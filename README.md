<div align="center">

# MechET

**Reason backward through executable electron flow; falsify forward with an independent compact expert**

[![Proof tests](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml)
[![Forward expert tests](https://github.com/wangyu-sd/MechET/actions/workflows/forward-expert-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/forward-expert-tests.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/RDKit-required-2E7D32?style=flat-square)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

[Architecture](#architecture) · [Frameworks](#framework-strategy) · [Quickstart](#quickstart) · [Training](#training-and-inference) · [Planning](#multistep-planning) · [Evaluation](#evaluation) · [Docs](#documentation)

</div>

---

## Central idea

MechET represents retrosynthesis as a chain of executable electron-flow
operations rather than an answer followed by a post-hoc explanation.

```text
mapped product
  -> inverse actor: electron-flow CoT / MECH_PROOF v1
  -> deterministic executor: formal hard gate
  -> executor-derived precursor candidates
  -> compact forward electron-flow expert
       - source/sink process score
       - target recovery
       - target-versus-competitor selectivity
  -> RL reward, hypothesis reranking and route-search edge cost
```

The project deliberately separates three kinds of evidence:

| Evidence | Question | Interpretation |
|---|---|---|
| Formal execution | Are maps, bonds, charges and electron accounting internally valid? | deterministic hard gate |
| Forward mechanistic evidence | Does a separately trained model support the forward moves and target over competitors? | calibrated soft evidence |
| Experimental feasibility | Will the reaction work with useful rate, selectivity and yield? | requires conditions, precedent, computation and/or experiment |

## Architecture

### Inverse actor

The inverse actor generates `MECH_PROOF v1` programs from an atom-mapped target.
`IMPORT`, `BOND`, `LP`, `CHARGE` and `EDGE` are **local executable primitives**,
not fixed complete reaction templates. The precursor is derived only by executing
the proof; there is no independent answer channel that can bypass the reasoning.

The long-term agent form is a small tool-using model that can:

1. inspect the current electron containers;
2. test explicit source-to-sink electron moves;
3. read deterministic state or failure feedback;
4. submit one complete inverse proof;
5. abstain when support is insufficient.

### Deterministic executor

For each proof edge the executor constructs the mapped state, applies bond and
charge changes, recomputes bond-electron and lone-pair deltas, checks electron
conservation, sanitizes the RDKit state, enforces chain/tree/DAG consistency and
derives the precursor.

### Compact forward expert

The independent forward expert is a small pure-PyTorch graph-pointer model with:

- graph message passing;
- electron-container embeddings;
- source and source-conditioned sink heads;
- precursor-product compatibility;
- a condition channel;
- target-versus-competitor contrastive training;
- uncertainty and route edge cost.

It is intentionally architecturally different from the inverse language model.
The current scope is mapped, closed-shell, two-electron polar chemistry. Radicals,
metal orbitals, spin states and photochemical one-electron pathways are reported
as unsupported rather than guessed.

### Generate–Falsify–Repair

Generate–Falsify–Repair remains available for complete proof programs:

```text
generate -> execute/falsify -> structured certificate -> repair or resample
```

A separate repair model is no longer required for the agent path: the same actor
can read a tool failure and choose a different action.

### K proof hypotheses

For one target, the same actor may be sampled repeatedly:

```text
pi_1, ..., pi_K ~ p_theta(proof | product)
```

K proof hypotheses are a sampling budget, not K stored templates. Candidates are
executed, deduplicated by partial-order equivalence and grouped by endpoint.
Primary set-valued metrics include `ExecutePass@K`, `EndpointPass@K`, unique
executable proof classes and unique structural precursor endpoints.

## Framework strategy

The chemistry environment is kept independent from fast-moving agent libraries.
`src/mechet/agent_env.py` owns molecular state, tools and rewards. Community
frameworks wrap that contract rather than reimplementing chemistry.

| Layer | Default | Role |
|---|---|---|
| Agent environment | `MechETAgentEnv` | shared state, tool calls, proof execution and rewards |
| Small-scale agent RL | Hugging Face TRL | reference `environment_factory` + GRPO implementation |
| Distributed agent RL | verl | preferred asynchronous multi-turn scale backend after the prototype is stable |
| Alternative scale backend | OpenRLHF | Ray/vLLM option for existing cluster stacks |
| Tracing and hierarchical credit | Agent Lightning | optional execution/training decoupling and observability |
| Reproducible environment packaging | Prime Verifiers | optional benchmark/evaluation adapter |
| Multistep planning | Syntheseus | default custom-model and matched-search benchmark |
| External template baseline | AiZynthFinder | template-policy MCTS comparison |

Only the TRL and Syntheseus adapters are implemented as first-class entrypoints
in this repository. verl, Agent Lightning, OpenRLHF and Prime Verifiers are
migration targets around the same environment contract, not parallel chemistry
implementations.

See [`docs/FRAMEWORK_MIGRATION.md`](docs/FRAMEWORK_MIGRATION.md) for the detailed
rationale, limitations and staged migration plan.

## Status

MechET is a research preview. The repository provides runnable infrastructure,
but no paper-scale trained checkpoint or frozen scientific result table is
claimed.

| Component | Status |
|---|---|
| `MECH_PROOF v1` compiler, executor and verifier | available |
| proof equivalence, diagnostics, GFR and hypothesis inference | available |
| compact forward electron-flow expert | available |
| data download, standardization, training, generation and evaluation | available |
| framework-neutral stateful agent environment | available |
| TRL agentic-GRPO reference adapter | available |
| Syntheseus offline candidate-pool adapter | available |
| online actor serving inside Syntheseus | planned after offline benchmark freeze |
| verl distributed agent loop | planned after TRL reward validation |
| public paper checkpoints and paper-scale results | not released |
| kinetic, transition-state or experimental validation | external evidence required |

## Quickstart

### Install

```bash
git clone https://github.com/wangyu-sd/MechET.git
cd MechET

# Core proof executor and tests
pip install -e ".[dev]"

# Compact forward expert
pip install -e ".[forward]"

# Data download and Arrow/Parquet processing
pip install -e ".[data]"

# Optional mapping and ORD decoding
pip install -e ".[mapping,ord]"

# Optional sequence-model baselines
pip install -e ".[baselines]"

# Stateful tool-use RL with TRL
pip install -e ".[agent]"

# Syntheseus route search
pip install -e ".[planning]"
```

### Execute one inverse proof

```python
from mechet.proof_program import (
    ChargeAction,
    ProofEdge,
    ProofProgram,
    format_proof_output,
    verify_proof,
)

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

result = verify_proof(
    format_proof_output(program),
    expected_precursor="[CH3:1][Br:3].[OH-:2]",
)
print(result["execute_ok"], result["endpoint_exact"])
```

### Inspect the stateful agent environment

```python
import json
from mechet.agent_env import MechETAgentEnv

env = MechETAgentEnv()
print(env.reset(target_smiles="[CH3:1][OH:2]"))
print(json.loads(env.inspect_state())["sources"])
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

The coupled arrows are applied atomically; the verifier does not require an
invalid pentavalent-carbon intermediate.

## Training and inference

### Forward expert data

```bash
# Inspect a registered source without downloading.
python scripts/forward_expert_data.py download \
  --dataset mech_uspto_31k --dry-run

# Standardize public mechanistic data.
python scripts/forward_expert_data.py standardize \
  --input data/raw/mech_uspto_31k \
  --output data/forward_expert/reactions.jsonl \
  --source mech_uspto_31k

python scripts/forward_expert_data.py build \
  --input data/forward_expert/reactions.jsonl \
  --output-dir data/forward_expert/steps
```

The standardizer never invents ambiguous source-sink labels. Invalid or unmapped
rows are quarantined; outcome-only rows may supervise product compatibility but
not pointer heads.

### Train the forward expert

```bash
python scripts/train_forward_expert.py \
  --config configs/forward/forward_expert_small.yaml
```

A CPU smoke configuration is provided in
`configs/forward/forward_expert_tiny.yaml`.

### Forward inference and generation

```bash
python scripts/run_forward_expert.py infer \
  --checkpoint outputs/forward_expert/small/best \
  --input data/forward_expert/steps/test.jsonl \
  --output outputs/forward_expert/test_predictions.jsonl \
  --auto-competitors 8

python scripts/run_forward_expert.py generate \
  --checkpoint outputs/forward_expert/small/best \
  --input data/forward_expert/steps/test.jsonl \
  --output outputs/forward_expert/generated_paths.jsonl \
  --beam-size 16 --max-steps 6 --stop-when-solved
```

### Train the small inverse tool-using actor

First validate the dataset and environment contract without importing TRL:

```bash
python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml \
  --dry-run --limit 8
```

Then launch agentic GRPO:

```bash
python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml
```

Recommended first models are Qwen3-0.6B and 1.7B. Freeze the compact forward
expert during actor RL. If the synchronous prototype is stable but tool latency
limits throughput, migrate the unchanged environment to verl rather than
rewriting the chemistry loop.

## Multistep planning

### Existing native search

`src/mechet/proof_routes.py` provides proof-carrying best-first search. Formally
invalid edges never enter the frontier; learned forward/selectivity evidence is
used as soft cost rather than irreversible pruning.

### Syntheseus benchmark adapter

Generate and optionally forward-rerank an offline hypothesis pool, then run a
matched-budget Retro* or breadth-first search:

```bash
python scripts/rerank_proof_hypotheses_forward.py \
  --predictions outputs/proof/hypotheses.jsonl \
  --checkpoint outputs/forward_expert/small/best \
  --output outputs/proof/hypotheses_forward_ranked.jsonl

python scripts/run_syntheseus_search.py \
  --candidate-pool outputs/proof/hypotheses_forward_ranked.jsonl \
  --targets data/benchmarks/paroutes/targets.smi \
  --inventory data/benchmarks/paroutes/inventory.smi \
  --output-dir outputs/planning/syntheseus_retrostar \
  --algorithm retro_star
```

Use offline pools for the first paper comparison so all planners receive the
same candidate set and reaction-model-call budget. Online actor expansion is a
separate experiment.

## Evaluation

Forward-expert evaluation:

```bash
python scripts/run_forward_expert.py eval \
  --predictions outputs/forward_expert/test_predictions.jsonl \
  --output outputs/forward_expert/test_metrics.json

python scripts/run_forward_expert.py eval-generation \
  --predictions outputs/forward_expert/generated_paths.jsonl \
  --output outputs/forward_expert/generation_metrics.json
```

Required families include:

- formal pass, false acceptance and false rejection;
- source/sink and complete move accuracy;
- target rank and target-versus-competitor margin;
- Brier score, calibration and risk-coverage;
- synthon and endpoint accuracy;
- family, temporal, scaffold and composition OOD;
- route solved rate, fully verified route rate, invalid edge rate, route length,
  diversity and matched search cost;
- abstention quality and verifier disagreement audits.

The authoritative paper contract remains
[`docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md`](docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md).

## Two-small-model learning schedule

Do not update the inverse actor and learned verifier simultaneously as a GAN.
Use alternating falsifier-guided training:

1. pretrain the inverse actor and forward expert separately;
2. freeze the forward expert and train the actor with formal process rewards and
   soft forward/selectivity terminal rewards;
3. freeze the actor and mine high-scoring verifier mistakes as hard negatives;
4. update and recalibrate the forward expert;
5. repeat for a small fixed number of audited rounds.

The deterministic executor is the permanent hard gate. Actor and verifier must
retain separate checkpoint lineage, data folds and fixed expert audit sets.

## Documentation

Start with [`docs/README.md`](docs/README.md).

- [`docs/PROOF_CARRYING.md`](docs/PROOF_CARRYING.md) — proof semantics and executor
- [`docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md`](docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md) — authoritative experiment contract
- [`docs/PROOF_EQUIVALENCE.md`](docs/PROOF_EQUIVALENCE.md) — equivalence and compositional OOD
- [`docs/DATA_LEAKAGE_AND_ICLR_PLAN.md`](docs/DATA_LEAKAGE_AND_ICLR_PLAN.md) — lineage and leakage controls
- [`docs/FORWARD_ELECTRON_EXPERT.md`](docs/FORWARD_ELECTRON_EXPERT.md) — forward expert specification
- [`docs/FRAMEWORK_MIGRATION.md`](docs/FRAMEWORK_MIGRATION.md) — TRL, verl, Agent Lightning, Verifiers and Syntheseus strategy
- [`data/FORWARD_EXPERT.md`](data/FORWARD_EXPERT.md) — local data/checkpoint layout

## Boundaries

- Formal executability is not kinetic, energetic or experimental validation.
- Forward and selectivity scores are learned soft evidence, not an infallible
  oracle.
- Selectivity requires explicit competitors; no competitor set means no valid
  selectivity claim.
- Atom-mapped inputs are required by the current executor and forward expert.
- Tool-call or JSON validity does not imply chemical validity.
- Third-party datasets, models and frameworks retain their upstream licenses.
