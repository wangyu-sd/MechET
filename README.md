<div align="center">

# MechET

**Bidirectional electron-flow reasoning for reliable retrosynthesis**

[![Proof tests](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml)
[![Forward expert tests](https://github.com/wangyu-sd/MechET/actions/workflows/forward-expert-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/forward-expert-tests.yml)
[![Agent framework tests](https://github.com/wangyu-sd/MechET/actions/workflows/agent-framework-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/agent-framework-tests.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/RDKit-required-2E7D32?style=flat-square)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

[Scientific story](#scientific-story) · [Method](#method) · [Current status](#current-status) · [Quickstart](#quickstart) · [Training](#training-and-inference) · [Planning](#multistep-planning) · [Evaluation](#evaluation) · [Documentation](#documentation)

</div>

---

## One-sentence contribution

MechET trains a small inverse actor to reason backward through **local executable primitives** of electron flow, derives precursors only by deterministic execution, and uses an architecturally independent compact forward expert to falsify whether those precursors preferentially recover the target rather than competing products.

```text
reason backward
  -> execute every electron-flow claim
  -> derive the precursor
  -> falsify forward
  -> retain, repair, abstain, or search further
```

## Scientific story

### The problem

Retrosynthesis systems face a persistent trade-off:

- reaction templates and named-reaction rules are reliable within their coverage but fragment chemical knowledge into isolated transformation classes;
- unconstrained graph or language generation is flexible but can hallucinate atom maps, reaction centers, reagents, mechanisms, or routes;
- endpoint-only evaluation cannot distinguish a chemically grounded prediction from an answer that bypasses its explanation;
- ordinary forward round-trip scoring checks only the final product and gives little credit assignment to the intermediate reasoning process.

### The hypothesis

Many apparently isolated reactions can be decomposed into a smaller vocabulary of reusable electron-flow operations such as bond-to-lone-pair movement, lone-pair-to-bond formation, pi-bond shifts, leaving-group departure, proton transfer, addition, elimination, and re-formation of unsaturation.

If these primitives are:

1. represented as executable actions;
2. composed by a tool-using inverse model;
3. checked after every step by a deterministic environment;
4. independently evaluated in the forward direction;
5. trained with process-level and terminal rewards;

then a smaller model may achieve better compositional generalization and lower hallucination rates than a larger direct-answer model.

### The ICLR claim under test

The paper does **not** rely on the claim that electron flow, chain-of-thought, forward models, or retrosynthesis planning are individually new. The contribution is their integration around one auditable computational object:

> electron-flow primitives are simultaneously the reasoning vocabulary, the tool actions, the formal verification units, the process-reward units, and the edges used in multistep search.

The central experimental questions are:

1. Does executable electron-flow CoT improve faithfulness and formal validity over outcome-only, free-form CoT, state-CoT, and net-edit baselines?
2. Does learning reusable primitives improve family, scaffold, temporal, and primitive-composition OOD generalization?
3. Can a compact forward expert provide useful process and selectivity evidence beyond ordinary endpoint round-trip scoring?
4. Can a small tool-using inverse actor match or outperform a larger direct generator under matched data and compute?
5. Do formally verified and forward-supported single-step edges improve multistep route reliability under matched search budgets?

The full collaborator-facing contract is in [`docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md`](docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md).

## Method

```text
atom-mapped target product
        |
        v
small inverse tool-using actor
  - inspect electron containers
  - propose source-to-sink moves
  - read execution feedback
  - compose a complete inverse proof
  - abstain when support is insufficient
        |
        v
deterministic executor
  - atom-map, bond, charge and electron checks
  - sanitizable state transitions
  - chain/tree/DAG consistency
  - executor-derived precursor
        |
        v
compact forward electron-flow expert
  - next source/sink compatibility
  - precursor-to-target compatibility
  - target-versus-competitor margin
  - uncertainty
        |
        v
ranking, process reward, repair/resampling, or route-search edge cost
```

### 1. Inverse actor

The inverse actor receives an atom-mapped product. The current compatibility path generates `MECH_PROOF v1`; the agent path can additionally inspect the molecular state and test explicit source-to-sink electron moves before submitting a complete proof.

A proof contains `IMPORT`, `BOND`, `LP`, `CHARGE`, and `EDGE` operations. These are local executable primitives rather than a library of complete reaction templates. The model cannot obtain endpoint credit through a separate answer channel: the precursor is produced only by execution.

The reference small-model path uses Qwen-family tool-calling models with TRL agentic GRPO. The first matched scale study uses approximately 0.6B, 1–2B, and 8B reference actors rather than assuming a larger model is always necessary.

### 2. Deterministic executor

The executor is never trained. For every proof edge it:

1. constructs the mapped molecular state and declared imports;
2. applies bond and formal-charge changes;
3. recomputes bond-electron and lone-pair deltas;
4. checks the written transition and exact electron conservation;
5. sanitizes the resulting RDKit state;
6. resolves chain, tree, and DAG dependencies;
7. rejects inconsistent joins or unreachable states;
8. returns the precursor state.

A learned score can never override an executor failure.

### 3. Stateful tool environment

`MechETAgentEnv` exposes the chemistry as a framework-neutral environment:

```text
inspect_state
apply_electron_move
apply_coupled_electron_moves
submit_proof
abstain
get_reward
```

The environment owns the molecule, electron-container inventory, tool budget, visited states, failure history, proof execution, process reward, terminal reward, and serializable rollout trace. TRL, verl, OpenRLHF, Agent Lightning, or other agent frameworks must wrap this contract rather than duplicate chemistry rules.

### 4. Compact forward electron-flow expert

The independent forward expert is intentionally smaller and architecturally different from the inverse language model. The default implementation is a pure-PyTorch graph model with:

- atom/bond message passing;
- electron-container embeddings;
- a source pointer head;
- a source-conditioned sink pointer head;
- a precursor-product compatibility head;
- a condition channel;
- target-versus-competitor contrastive training;
- uncertainty and route-cost outputs.

It supports mapped, closed-shell, two-electron polar chemistry in v1. Radicals, metal orbitals, spin-state changes, coordination chemistry, and photochemical one-electron processes are reported as unsupported rather than guessed.

The forward expert is learned **soft evidence**, not a hard chemistry oracle. Selectivity is only meaningful when explicit competing sites, products, mechanisms, or stereochemical outcomes are included.

### 5. Generate–Falsify–Repair

Generate–Falsify–Repair remains the compatibility path for complete proof programs:

```text
generate
  -> deterministic falsification
  -> structured failure certificate
  -> local repair, agent revision, or resampling
```

In the agent path, the same actor can read a failed tool result and choose a different move; an independent Repair Actor is retained only as a controlled baseline.

### 6. K proof hypotheses

For one product, the same actor may be sampled repeatedly:

```text
pi_1, ..., pi_K ~ p_theta(proof | product)
```

K proof hypotheses are a sampling and test-time-compute budget, not K stored reaction templates. Candidates are executed, deduplicated by partial-order equivalence, grouped by structural endpoint, and optionally reranked with forward evidence.

Set-valued metrics include `ExecutePass@K`, `EndpointPass@K`, unique executable proof classes, unique mechanism compositions, and unique structural precursor endpoints.

### 7. Alternating two-small-model learning

The inverse actor and forward expert are not trained simultaneously as a GAN. The audited schedule is:

1. pretrain the inverse actor and forward expert independently;
2. freeze the forward expert and improve the actor with formal process rewards plus soft forward/selectivity terminal rewards;
3. freeze the actor and mine high-scoring actor–verifier disagreements;
4. send disagreements to an audit set rather than treating them as negatives;
5. update and recalibrate the forward expert only on independently verified negatives;
6. repeat for a small fixed number of rounds.

Accepted negative evidence must come from expert review, experiment, a known competing product, or an independently calibrated ensemble. A prediction that differs from one patent record is not automatically chemically wrong.

### 8. Multistep planning and reaction networks

A verified single-step proposal becomes a proof-carrying route edge. The repository provides:

- native best-first route search;
- a Syntheseus backward-model adapter;
- matched-budget Retro* and breadth-first experiments;
- offline candidate pools for deterministic planner comparisons;
- route-level formal verification and building-block termination;
- reaction-network construction from surviving proof hypotheses.

Formal invalidity is a hard prune. Learned forward, selectivity, precedent, and uncertainty scores are soft ranking terms unless their false-rejection rates have been calibrated for the relevant reaction family.

## What is implemented

- `MECH_PROOF v1` compiler, parser, executor, verifier, equivalence, diagnostics, and structured failure certificates;
- proof-SFT, Verifier-DPO, proof-set RLVR, bounded repair, K-hypothesis inference, and Generate–Falsify–Repair;
- conservative data normalization and decontamination utilities;
- source/sink electron-container representation and coupled-arrow execution;
- compact graph-pointer forward expert with training, inference, generation, calibration, and selectivity scoring;
- framework-neutral `MechETAgentEnv`;
- TRL agentic-GRPO reference training for small inverse actors;
- optional scale migration to verl/OpenRLHF without changing chemistry semantics;
- audit-first actor–verifier disagreement mining and verified-negative fine-tuning;
- native route search and Syntheseus planning adapters;
- dedicated proof, forward, agent, and documentation CI workflows.

## Current status

MechET is a research preview. Infrastructure is available; paper-scale checkpoints and frozen scientific result tables are not yet released.

| Component | Status |
|---|---|
| Deterministic proof executor and formal verifier | available |
| Proof SFT/DPO/RLVR and K-hypothesis inference | available |
| Source-to-sink tool environment | available |
| Compact forward expert pipeline | available |
| TRL small-actor reference adapter | available |
| Syntheseus offline planning adapter | available |
| Data download, standardization, manifests, and quarantine | available |
| Paper-scale forward checkpoint | not released |
| Paper-scale inverse checkpoints | not released |
| Alternating actor–verifier performance results | not released |
| Matched multistep benchmark results | not released |
| Kinetic, transition-state, or experimental validation | external evidence required |

## Quickstart

### Install

```bash
git clone https://github.com/wangyu-sd/MechET.git
cd MechET

# Core executor and proof tests
pip install -e ".[dev]"

# Forward expert
pip install -e ".[forward]"

# Data download and Arrow/Parquet support
pip install -e ".[data]"

# Optional atom mapping and ORD decoding
pip install -e ".[mapping,ord]"

# Small inverse actor with TRL
pip install -e ".[agent]"

# Multistep planning with Syntheseus
pip install -e ".[planning]"
```

### Execute an inverse proof

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

### Inspect the electron-flow environment

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

Coupled arrows are applied atomically, so the verifier does not require an artificial pentavalent-carbon intermediate.

## Training and inference

### 1. Download and standardize forward data

```bash
python scripts/forward_expert_data.py download \
  --dataset mech_uspto_31k --dry-run

python scripts/forward_expert_data.py standardize \
  --input data/raw/mech_uspto_31k \
  --output data/forward_expert/reactions.jsonl \
  --source mech_uspto_31k

python scripts/forward_expert_data.py build \
  --input data/forward_expert/reactions.jsonl \
  --output-dir data/forward_expert/steps
```

The standardizer never invents an ambiguous arrow label. Unmapped or invalid rows are quarantined; outcome-only rows may supervise reaction compatibility but not electron source/sink heads.

### 2. Build and audit inverse proof data

```bash
python scripts/build_mechet_sft.py \
  --flower-root /path/to/flower_new_dataset \
  --out-dir data/mechet_sft \
  --splits train valid test

python scripts/build_mechet_proof_sft.py \
  --input-dir data/mechet_sft \
  --output-dir data/mechet_proof_sft \
  --splits train valid test

python scripts/audit_reaction_overlap.py \
  --train data/mechet_proof_sft/train.jsonl \
  --benchmark data/benchmarks/uspto50k/test.csv \
  --benchmark-format reaction_table \
  --reaction-field reaction_smiles \
  --out-dir outputs/data_audit/flower_vs_uspto50k_test
```

### 3. Train the compact forward expert

```bash
python scripts/train_forward_expert.py \
  --config configs/forward/forward_expert_small.yaml
```

A CPU smoke configuration is available at `configs/forward/forward_expert_tiny.yaml`.

### 4. Train the inverse proof baselines

```bash
python scripts/train_mechet_sft.py \
  --config configs/proof/proof_actor_sft.yaml

python scripts/train_proof_dpo.py \
  --config configs/proof/proof_dpo.yaml
```

These complete-proof models remain required baselines even when the main method uses tool-integrated reasoning.

### 5. Train the small inverse tool-using actor

Validate the dataset and environment without loading a model:

```bash
python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml \
  --dry-run --limit 8
```

Launch agentic GRPO:

```bash
python scripts/train_inverse_agent_trl.py \
  --config configs/agent/inverse_trl_grpo.yaml
```

Freeze the forward expert during each actor-RL phase. Migrate the unchanged environment to verl only after the synchronous TRL experiment has stable reward decomposition and passes reward-hacking audits.

### 6. Generate and rerank K hypotheses

```bash
python scripts/infer_proof_hypotheses.py \
  --data data/mechet_proof_clean/test.jsonl \
  --adapter outputs/proof/actor/adapter \
  --samples-per-target 64 \
  --out outputs/proof/hypotheses.jsonl

python scripts/rerank_proof_hypotheses_forward.py \
  --predictions outputs/proof/hypotheses.jsonl \
  --checkpoint outputs/forward_expert/small/best \
  --output outputs/proof/hypotheses_forward_ranked.jsonl
```

### 7. Mine audited actor–verifier disagreements

```bash
python scripts/mine_bidirectional_hard_negatives.py \
  --predictions outputs/proof/hypotheses_forward_ranked.jsonl \
  --checkpoint outputs/forward_expert/small/best \
  --output data/forward_expert/audit_candidates.jsonl
```

The output is an audit queue, not a negative training set. After independent review and explicit label provenance:

```bash
python scripts/train_forward_expert_hard_negative.py \
  --checkpoint outputs/forward_expert/small/best \
  --positive-data data/forward_expert/steps/train.jsonl \
  --negative-data data/forward_expert/verified_negatives.jsonl \
  --output outputs/forward_expert/adversarial_round1
```

## Multistep planning

### Native proof-carrying search

`src/mechet/proof_routes.py` admits only executor-verified edges. The learned forward expert supplies soft edge costs and uncertainty, not an irreversible hard gate.

### Syntheseus matched-budget planning

```bash
python scripts/run_syntheseus_search.py \
  --candidate-pool outputs/proof/hypotheses_forward_ranked.jsonl \
  --targets data/benchmarks/paroutes/targets.smi \
  --inventory data/benchmarks/paroutes/inventory.smi \
  --output-dir outputs/planning/syntheseus_retrostar \
  --algorithm retro_star
```

The first paper comparison uses frozen offline candidate pools so all planners receive the same one-step candidates, stock, reaction-model-call budget, iteration budget, and wall-clock budget. Online actor expansion is reported separately.

## Evaluation

The ICLR evaluation is organized around five evidence layers.

| Layer | Primary metrics |
|---|---|
| Endpoint | structural precursor Top-1/5/10, reaction-center, synthon accuracy |
| Process | source/sink accuracy, complete move accuracy, proof equivalence, execution rate |
| Reliability | false acceptance rate, false rejection rate, calibration, risk–coverage, abstention |
| Generalization | family, scaffold, temporal, primitive-composition, and proof-topology OOD |
| Planning | solved rate, fully verified route rate, invalid edge rate, route length/diversity, search cost |

Mandatory set-valued metrics include `ExecutePass@K` and `EndpointPass@K`. Mandatory selectivity reporting includes target rank, target-versus-best-competitor margin, competitor-set construction, and per-family calibration.

No single Top-1 number is sufficient for the paper claim.

## ICLR result map

The planned paper is organized around four claims.

### R1 — Electron-flow primitives are reusable and compositional

- primitive-seen/composition-unseen split;
- family, scaffold, temporal, and reaction-center complexity strata;
- comparison with direct generation, reaction-center-only, net-edit, state-CoT, and free-form CoT;
- primitive and sequence accuracy, synthon accuracy, and endpoint accuracy.

### R2 — Executable tool-grounded CoT reduces hallucination

- formal executor pass rate;
- answer–reasoning disagreement for answer-bearing baselines;
- controlled-corruption false acceptance rate and error localization;
- recovery after tool failure;
- ablation of free-form CoT versus tool-interleaved CoT.

### R3 — Forward falsification improves reliability beyond endpoint round trip

- ordinary forward product scorer versus forward electron-flow expert;
- endpoint-only reward versus step-level process reward;
- target recovery, complete mechanism support, selectivity pair accuracy, calibration, and risk–coverage;
- adversarial actor–verifier audit set.

### R4 — Small bidirectional models improve multistep planning

- inverse actor scale study;
- matched candidate and search budgets;
- native best-first, Syntheseus Retro*, breadth-first, and external template baseline;
- solved rate, fully verified route rate, route confidence, invalid edges, nodes expanded, and diversity.

## Documentation

Start with [`docs/README.md`](docs/README.md).

- [`docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md`](docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md) — authoritative ICLR collaboration and experiment contract
- [`docs/PROOF_CARRYING.md`](docs/PROOF_CARRYING.md) — proof language and deterministic executor
- [`docs/FORWARD_ELECTRON_EXPERT.md`](docs/FORWARD_ELECTRON_EXPERT.md) — forward expert data, architecture, training, and evaluation
- [`docs/FRAMEWORK_MIGRATION.md`](docs/FRAMEWORK_MIGRATION.md) — TRL, verl, Agent Lightning, Prime Verifiers, and Syntheseus strategy
- [`docs/PROOF_EQUIVALENCE.md`](docs/PROOF_EQUIVALENCE.md) — partial-order equivalence and compositional OOD
- [`docs/DATA_LEAKAGE_AND_ICLR_PLAN.md`](docs/DATA_LEAKAGE_AND_ICLR_PLAN.md) — data lineage and benchmark leakage controls
- [`data/FORWARD_EXPERT.md`](data/FORWARD_EXPERT.md) — local data and checkpoint layout

## Boundaries

- Formal executability is not evidence of a low activation barrier, favorable kinetics, high yield, or experimental success.
- Forward and selectivity scores are learned soft evidence, not an infallible oracle.
- Selectivity requires explicit competitors; without a competitor set there is no valid selectivity claim.
- The current executor and forward expert require atom-mapped inputs.
- The current explicit source/sink environment supports closed-shell two-electron chemistry, not all reaction mechanisms.
- Tool-call or JSON validity does not imply chemical validity.
- An alternative executable endpoint is not a negative merely because it differs from a single recorded patent route.
- Third-party datasets, models, and frameworks retain their upstream licenses.
