<div align="center">

# MechET

**Reason backward through executable electron flow; verify forward with an independent compact expert**

[![Proof tests](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml)
[![Forward expert tests](https://github.com/wangyu-sd/MechET/actions/workflows/forward-expert-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/forward-expert-tests.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/RDKit-required-2E7D32?style=flat-square)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

[Architecture](#architecture) · [Quickstart](#quickstart) · [Forward expert](#compact-forward-electron-flow-expert) · [Data and training](#end-to-end-forward-expert-pipeline) · [Evaluation](#evaluation) · [Documentation](#documentation)

</div>

---

## Central idea

MechET treats a retrosynthetic proposal as a falsifiable electron-flow program.
The inverse actor cannot bypass its reasoning by emitting an unrelated answer:
the precursor is derived by deterministic execution of the generated proof.
A separate compact forward expert then asks whether those precursors preferentially
recover the target rather than competing products.

```text
mapped product
  -> inverse LLM actor: electron-flow CoT / MECH_PROOF v1
  -> deterministic executor: formal hard gate
  -> executor-derived precursor candidates
  -> compact forward electron-flow expert
       - next source/sink compatibility
       - target-product recovery
       - target-versus-competitor selectivity
  -> verified ranking, RL reward and route-search edge score
```

The project therefore separates three meanings that must not be conflated:

| Evidence | Question | Status |
|---|---|---|
| Formal execution | Are atom maps, bonds, charges and electron accounting internally valid? | deterministic hard gate |
| Forward mechanistic evidence | Does an independently trained forward model support the electron moves and target? | learned, calibrated soft evidence |
| Experimental feasibility | Will the reaction work with useful rate, selectivity and yield under real conditions? | requires conditions, precedent, computation and/or experiment |

## Architecture

### Inverse actor

The Qwen-family actor receives a mapped product and generates one or more
`MECH_PROOF v1` programs. `IMPORT`, `BOND`, `LP`, `CHARGE` and `EDGE` are
local executable primitives rather than a library of fixed whole-reaction templates.
The current representation records bond-electron redistribution; explicit
source-to-sink tool calls can be compiled to the same executor representation.

### Deterministic executor

For each proof edge, the executor:

1. constructs the mapped molecular state;
2. applies bond and charge operations;
3. recomputes bond-electron and lone-pair changes;
4. verifies written operations and electron conservation;
5. sanitizes the resulting RDKit state;
6. enforces reachability and consistent chain/tree/DAG joins;
7. derives the precursor without a separate answer channel.

### Compact forward electron-flow expert

The forward expert is intentionally smaller and architecturally different from
the inverse LLM. The default implementation is a pure-PyTorch graph model with:

- atom/bond message passing;
- electron-container embeddings;
- a source pointer head;
- a source-conditioned sink pointer head;
- a precursor-product compatibility head;
- a deterministic hashed condition channel for sparse solvent/reagent metadata;
- contrastive target-versus-competitor training.

It supports mapped, closed-shell, two-electron polar chemistry in v1. Coupled
arrows in one elementary event are applied atomically. Radicals, metal orbitals,
spin states, coordination and photochemical one-electron chemistry are out of
scope rather than guessed.

### Generate–Falsify–Repair

Generate–Falsify–Repair remains available for complete proof programs:

```text
generate candidate proof
  -> execute and falsify
  -> return structured failure certificate
  -> repair or resample
```

The forward expert adds an independent second direction. A proof can be formally
executable yet receive weak target recovery, low selectivity margin or high
uncertainty.

### K proof hypotheses

For a target product, the same autoregressive actor may be sampled repeatedly:

```text
pi_1, ..., pi_K ~ p_theta(proof | product)
```

K proof hypotheses are a sampling/search budget, not K stored reaction templates.
Candidates are executed, deduplicated by partial-order equivalence and grouped by
structural endpoint. Core set-valued metrics include `ExecutePass@K`,
`EndpointPass@K`, unique executable proof classes and unique precursor endpoints.
The compact forward expert can rerank the surviving endpoints without weakening
the formal gate.

## Status

MechET is a research preview. The code now includes an end-to-end forward-expert
pipeline, but no paper-scale trained checkpoint or frozen scientific result table
is claimed in this repository.

| Component | Status |
|---|---|
| `MECH_PROOF v1` compiler, executor and verifier | available |
| proof equivalence, diagnostics, GFR and proof-set inference | available |
| proof-carrying multistep search scaffold | available |
| forward source-sink formal step executor | available |
| compact graph-pointer forward expert | available |
| data download/standardization/build scripts | available |
| model pre-download, training, inference, generation and evaluation | available |
| public paper checkpoint and paper-scale metrics | not released |
| kinetic, transition-state or experimental validation | external evidence required |

## Quickstart

### Install

```bash
git clone https://github.com/wangyu-sd/MechET.git
cd MechET

# Proof executor and tests
pip install -e ".[dev]"

# Forward expert training
pip install -e ".[forward]"

# Dataset download/Arrow/Parquet support
pip install -e ".[data]"

# Optional atom mapping and ORD protobuf decoding
pip install -e ".[mapping,ord]"

# Optional ChemBERTa/MoLFormer/Qwen baselines
pip install -e ".[baselines]"
```

### Execute a proof

```python
from mechet import ChargeAction, ProofEdge, ProofProgram
from mechet import format_proof_output, verify_proof

program = ProofProgram(
    target_smiles="[CH3:1][OH:2]",
    roots={"s0": ["[Br-:3]"]},
    precursor_state_id="s1",
    edges=[ProofEdge(
        "s0", "s1",
        bonds=[(1, 2, -1), (1, 3, +1)],
        lone_pairs=[(2, +2), (3, -2)],
        charges=[
            ChargeAction(2, 0, -1),
            ChargeAction(3, -1, 0),
        ],
    )],
)

proof = format_proof_output(program)
result = verify_proof(
    proof,
    expected_precursor="[CH3:1][Br:3].[OH-:2]",
)
print(result["execute_ok"], result["endpoint_exact"])
```

### Verify coupled forward arrows

```python
from mechet.forward_expert import verify_electron_step

result = verify_electron_step(
    "[CH3:1][Br:3].[OH-:2]",
    [
        {
            "source": {"kind": "LP", "atoms": [2]},
            "sink": {"kind": "BOND", "atoms": [1, 2]},
        },
        {
            "source": {"kind": "BOND", "atoms": [1, 3]},
            "sink": {"kind": "ATOM", "atoms": [3]},
        },
    ],
)
print(result)
```

The two arrows are applied as one elementary event; the verifier does not require
an invalid pentavalent-carbon intermediate.

## Compact Forward Electron-Flow Expert

The expert has two complementary outputs.

### Process score

For current state `s`, it assigns:

```text
p(source | s)
p(sink | source, s)
```

This permits step-level process rewards and state-level branching during
agent/tool-use training.

### Outcome and selectivity score

For precursors `R`, target `P` and competitor set `C`:

```text
S(R, P)
margin = S(R, P) - max_{P' in C} S(R, P')
```

The target score alone is not a selectivity guarantee. The competitor policy
must enumerate alternative sites, regio/stereoisomers and relevant side products,
and thresholds must be calibrated by reaction family.

## End-to-end forward expert pipeline

### 1. Inspect and download data

Registered sources and licenses are documented in
`configs/forward/data_sources.yaml`.

```bash
# No network call: show what would be downloaded.
python scripts/forward_expert_data.py download \
  --dataset mech_uspto_31k --dry-run

# Public source with frozen revision and SHA-256 manifest.
python scripts/forward_expert_data.py download \
  --dataset mech_uspto_31k \
  --revision <commit-or-tag> \
  --output data/raw

python scripts/forward_expert_data.py inspect \
  --input data/raw/mech_uspto_31k
```

Restricted sources are not fetched unless the caller explicitly acknowledges the
upstream terms with `--accept-restricted-license`. The script records provenance;
it does not grant redistribution rights.

### 2. Standardize reactions and electron-flow labels

```bash
python scripts/forward_expert_data.py standardize \
  --input data/raw/mech_uspto_31k \
  --output data/forward_expert/reactions.jsonl \
  --source mech_uspto_31k

# Outcome-only sources such as ORD can first be standardized without maps,
# then mapped with RXNMapper before building training examples.
python scripts/forward_expert_data.py standardize \
  --input data/raw/ord_data \
  --output data/forward_expert/ord_unmapped.jsonl \
  --source ord_data \
  --allow-unmapped

python scripts/forward_expert_data.py map \
  --input data/forward_expert/ord_unmapped.jsonl \
  --output data/forward_expert/ord_mapped.jsonl

python scripts/forward_expert_data.py build \
  --input data/forward_expert/reactions.jsonl \
  --output-dir data/forward_expert/steps
```

The standardizer directly decodes the MechSMILES columns used by the public
FlowER/mech-USPTO mirrors and the arrow-code convention used by PMechDB. ORD
protobuf rows are decoded through the official `ord-schema`; RXNMapper is an
optional, explicit mapping stage for outcome-only unmapped records.

The conservative normalizer never invents a source-sink arrow. Unmapped or
invalid rows are quarantined. Rows with reaction outcomes but no unambiguous
arrows may train the reaction compatibility head but not the move pointer heads.

### 3. Pre-download alternative models

```bash
python scripts/forward_expert_data.py predownload \
  --model chemberta \
  --model molformer \
  --revision <commit-or-tag> \
  --output models/baselines
```

`qwen_small` is available as a sequence-model ablation. The graph expert remains
the default because it is independent from the inverse Qwen actor and efficient
enough for repeated RL and route-search scoring.

### 4. Train

```bash
python scripts/train_forward_expert.py \
  --config configs/forward/forward_expert_small.yaml
```

A tiny CPU smoke configuration is included:

```bash
python scripts/train_forward_expert.py \
  --config configs/forward/forward_expert_tiny.yaml \
  --device cpu
```

Training combines source CE, source-conditioned sink CE, positive/negative
reaction compatibility and target-versus-competitor margin loss. `best/` and
`last/` checkpoints, metadata and JSONL logs are saved under the configured
output directory.

### 5. Inference

```bash
python scripts/run_forward_expert.py infer \
  --checkpoint outputs/forward_expert/small/best \
  --input data/forward_expert/steps/test.jsonl \
  --output outputs/forward_expert/test_predictions.jsonl \
  --auto-competitors 8
```

`--auto-competitors` adds formally reachable alternative states when an explicit
competitor list is unavailable. This is a bounded negative-generation policy,
not a claim that every experimental side product has been enumerated.

### 6. Forward electron-flow generation

```bash
python scripts/run_forward_expert.py generate \
  --checkpoint outputs/forward_expert/small/best \
  --input data/forward_expert/steps/test.jsonl \
  --output outputs/forward_expert/generated_paths.jsonl \
  --beam-size 16 \
  --branch-limit 24 \
  --proposal-pool 48 \
  --max-steps 6 \
  --stop-when-solved
```

Generation considers formally executable single-arrow and locally coupled
two-arrow events. Formal execution is a hard filter; learned product scores are
optional soft reranking signals.

## Evaluation

```bash
python scripts/run_forward_expert.py eval \
  --predictions outputs/forward_expert/test_predictions.jsonl \
  --output outputs/forward_expert/test_metrics.json

python scripts/run_forward_expert.py eval-generation \
  --predictions outputs/forward_expert/generated_paths.jsonl \
  --output outputs/forward_expert/generation_metrics.json
```

Implemented metrics include:

- formal pass, false acceptance and false rejection when labels exist;
- next electron-move Top-1 and reciprocal rank;
- target-product Top-1;
- target-versus-competitor selectivity support;
- Brier score and expected calibration error.

A paper evaluation must additionally include patent/time/family holdouts,
mechanism-family stratification, reaction complexity, multiple reaction centres,
risk-coverage curves, abstention and fully verified route rate under matched
search budgets. The authoritative proof experiment contract remains
[`docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md`](docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md).

## Existing proof and route pipelines

The original proof actor, DPO/RLVR/GFR inference and route-search scripts remain
available. The forward expert is an additive verifier rather than a replacement
for the deterministic executor.

Use `MECHET_FORWARD_EXPERT_PATH` to expose a trained checkpoint through the
existing plausibility oracle interface:

```bash
export MECHET_FORWARD_EXPERT_PATH=outputs/forward_expert/small/best
export MECHET_FORWARD_EXPERT_DEVICE=cpu
```

```python
from mechet.plausibility import load_oracle
oracle = load_oracle("mechet.forward_oracle:score_payload")
```

Existing proof-hypothesis files can be reranked directly:

```bash
python scripts/rerank_proof_hypotheses_forward.py \
  --predictions outputs/proof/hypotheses.jsonl \
  --checkpoint outputs/forward_expert/small/best \
  --output outputs/proof/hypotheses_forward_ranked.jsonl
```

`mechet.forward_rewards.score_inverse_proof_forward` exposes a complete-proof
reward for RL code. It executes the inverse proof first and then scores the
derived precursor in the independent forward direction.

For multistep planning, formally invalid edges are rejected. Forward target
recovery, selectivity margin and uncertainty are initially used as soft route
costs, not irreversible learned pruning decisions.

## Documentation

Start with [`docs/README.md`](docs/README.md).

- [`docs/PROOF_CARRYING.md`](docs/PROOF_CARRYING.md) — proof semantics and deterministic executor
- [`docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md`](docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md) — authoritative proof experiment contract
- [`docs/PROOF_EQUIVALENCE.md`](docs/PROOF_EQUIVALENCE.md) — partial-order equivalence and compositional OOD
- [`docs/DATA_LEAKAGE_AND_ICLR_PLAN.md`](docs/DATA_LEAKAGE_AND_ICLR_PLAN.md) — data lineage and leakage controls
- [`docs/FORWARD_ELECTRON_EXPERT.md`](docs/FORWARD_ELECTRON_EXPERT.md) — forward expert data, model, training, inference, generation and evaluation
- [`data/FORWARD_EXPERT.md`](data/FORWARD_EXPERT.md) — local data/checkpoint layout

## Boundaries

- Formal executability is not energetic, kinetic or experimental validation.
- The forward expert is a calibrated learned verifier, not an infallible oracle.
- Selectivity requires an explicit competitor set; no competitor set means no
  meaningful selectivity claim.
- Atom-mapped inputs are required by the current executor and forward expert.
- `MECH_PROOF v1` remains the stable low-level proof format; source-sink actions
  are compiled/verified against the same molecular-state semantics.
- Downloaded datasets and third-party checkpoints must follow their upstream
  licenses and must not be committed to this repository.
