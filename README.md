<div align="center">

<img src="docs/assets/mechet-icon.svg" alt="MechET project icon" width="168" />

# MechET

**Causal and compositional electron-flow reasoning for retrosynthesis**

[![Proof tests](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml)
[![Runtime tests](https://github.com/wangyu-sd/MechET/actions/workflows/agent-framework-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/agent-framework-tests.yml)
[![Evidence tests](https://github.com/wangyu-sd/MechET/actions/workflows/knowledge-ablation-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/knowledge-ablation-tests.yml)

[Research thesis](#research-thesis) · [Main method](#main-method) · [Experimental program](#experimental-program) · [Reproduce](#minimum-credible-reproduction) · [Documentation](#documentation)

</div>

---

> **MechET asks whether a mechanistic rationale can be the computation that determines a retrosynthetic prediction—not merely a plausible explanation written after the answer.**

## Research thesis

Most mechanism-aware language models optimize an answer and a rationale jointly, but do not guarantee that the rationale is causally upstream of the answer. A chemically plausible chain of thought may therefore coexist with an endpoint produced through an independent, unverifiable channel.

MechET removes that ambiguity by formulating retrosynthesis as **causal program induction over explicit source-to-sink electron-flow actions**. The model commits actions to a deterministic environment; an environment-owned trace is then the only computational source of the executable proof and precursor.

The study is organized around three falsifiable hypotheses:

| Hypothesis | Scientific question | Required evidence | Falsifying outcome |
|---|---|---|---|
| **H1 — causal faithfulness** | Does the committed tool trace determine the endpoint? | Successful `finish_trace`, trace/proof replay, and paired observation interventions under an identical runtime contract | Endpoint performance is unchanged when chemically relevant tool observations are removed, stale, or shuffled |
| **H2 — compositional basis** | Can familiar local execution primitives form unseen complete mechanisms? | Primitive-seen/composition-unseen splits built from `source_to_sink_execution_moves_v1` | Held-out examples contain unseen primitives, or performance is explained only by scaffold/template overlap |
| **H3 — evidence separation** | Does external mechanistic evidence improve program induction beyond extra context alone? | Frozen textbook/anchor conditions, length-matched controls, direct open-book controls, and zero evidence reward | Gains disappear against irrelevant context or arise from query leakage, runtime mismatch, or missing predictions |

The authoritative claim definitions and boundaries are in [`docs/SCIENTIFIC_THESIS.md`](docs/SCIENTIFIC_THESIS.md).

## Main method

```text
atom-mapped product
  -> inspect state and import missing mapped fragments
  -> explicit source-to-sink electron-flow actions
  -> environment-owned molecular-state transitions
  -> committed move trace
  -> finish_trace
  -> replay declared moves
  -> deterministic MECH_PROOF v1 compilation
  -> proof execution
  -> full_precursor_state
  -> structural_precursor + auxiliary_fragments
```

The main TRL-facing environment exposes only:

```text
inspect_state
import_fragment
apply_electron_move
apply_coupled_electron_moves
finish_trace
abstain
```

`reset` and `get_reward` are framework methods. Internal helpers such as `state_dict` and `_snapshot` remain private, and the main method does not expose `submit_proof`. Independent complete-proof submission is retained only as a named legacy baseline.

### Causal contract

A trace-owned prediction receives endpoint credit only when all of the following hold:

1. the model explicitly calls `finish_trace`;
2. the episode is finalized by the environment;
3. declared source-to-sink moves replay to the recorded states;
4. the compiled proof executes deterministically;
5. trace, move-sequence, and proof digests agree with the terminal result.

The evaluator never completes an unfinished trace on the model's behalf and never falls back from a trace condition to a free-form `PRECURSOR:` answer.

### Endpoint contract

Every executable result separates three views:

| Field | Meaning | Role |
|---|---|---|
| `full_precursor_state` | Complete executor-derived state, including imported and auxiliary species | Formal execution audit |
| `structural_precursor` | Fragments containing atoms originating from the target | Primary endpoint metric |
| `auxiliary_fragments` | Remaining salts, leaving groups, or mapped auxiliary species | Secondary analysis |

Primary structural accuracy ignores atom-map labels. Mapped exact match is reported separately.

### Evidence boundary

Textbook passages, structured mechanistic knowledge anchors, and learned forward scores are **soft evidence**. They may guide or rank an executable proposal, but they cannot:

- define formal validity;
- return or directly reward a precursor;
- override executor failure;
- establish kinetics, yield, condition compatibility, or experimental success.

## Terminology

- **Electron-flow execution primitive** — a local executable source-to-sink action such as `LP -> BOND`, `BOND -> ATOM`, or `BOND -> BOND`. These actions define the H2 composition basis.
- **Mechanistic knowledge anchor** — a provenance-aware retrieval record containing role bindings, candidate moves, preconditions, warnings, competitors, and references. Anchor IDs do not define H2.
- **MECH_PROOF v1** — a deterministic bond/lone-pair/charge program compiled from the move-bound trace in the main method. Independently generated proofs are baselines.
- **Trace-owned prediction** — a prediction whose only admissible endpoint is produced by `finish_trace` from committed environment transitions.

## Experimental program

The paper-level logic is deliberately sequential:

```text
coverage and leakage audit
  -> replay-verified Tool-SFT
  -> H1 causal intervention test
  -> H2 composition-OOD test
  -> H3 matched evidence test
  -> optional scale, RL, forward evidence, and planning
```

Later stages cannot rescue a failed earlier claim. In particular, planning quality does not establish trace faithfulness, and retrieval gains do not establish compositional generalization.

### H1 — causal faithfulness

Compare the normal trace-owned path against:

```text
remove_tool_observations
stale_tool_observations
shuffle_tool_observations
disable_inspect_state
disable_intermediate_execution
```

All artifacts must share the same model, adapter, frozen revision, seed policy, candidate count, temperature, top-p, token budget, iteration budget, and reference IDs.

### H2 — compositional generalization

The headline split holds out complete move compositions while requiring every constituent execution primitive to appear in training at a declared minimum frequency.

```text
primitive_basis = source_to_sink_execution_moves_v1
zero train/test complete-composition overlap
all test primitives represented in train
non-empty held-out test
```

Scaffold, family, reaction-center, and near-duplicate overlap must be reported separately rather than conflated with composition novelty.

### H3 — evidence separation

The matched conditions are:

| Condition | Endpoint path | Evidence |
|---|---|---|
| `trace_no_knowledge` | Trace-owned | None |
| `trace_length_matched_irrelevant` | Trace-owned | Irrelevant text with the same character budget |
| `trace_textbook_rag` | Trace-owned | Frozen textbook evidence |
| `trace_structured_anchors` | Trace-owned | Frozen structured anchors |
| `trace_text_plus_anchors` | Trace-owned | Both evidence layers |
| `direct_textbook_rag` | Direct answer | The same bounded textbook evidence |

The main retrieval query uses only inference-available molecular-state terms. `label_oracle` is an explicitly named upper bound and is not headline-eligible.

## Minimum credible reproduction

The complete operational sequence and stopping gates are in [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md). The commands below are the shortest path to a scientifically interpretable pilot.

### 1. Install

```bash
git clone https://github.com/wangyu-sd/MechET.git
cd MechET

pip install -e ".[dev]"
pip install -e ".[agent,knowledge]"
```

The released compatibility window includes `trl>=1.8,<2`, `transformers>=5.2,<6`, and `datasets>=4.7,<6`.

### 2. Build replay-verified trajectories

```bash
python scripts/build_textbook_tool_sft.py \
  --input data/mechet_proof_clean/train.jsonl \
  --corpus knowledge/corpus/passages.jsonl \
  --output data/textbook_tool_sft/train.jsonl \
  --query-mode state
```

Rows exceeding the frozen headline tool budget are quarantined rather than trained under a larger hidden budget.

### 3. Build and validate matched conditions

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml

python scripts/validate_experiment_contract.py \
  --model-name Qwen/Qwen3-0.6B \
  --condition trace_no_knowledge=data/knowledge_ablation/v2/trace_no_knowledge.jsonl \
  --condition trace_textbook_rag=data/knowledge_ablation/v2/trace_textbook_rag.jsonl \
  --condition direct_textbook_rag=data/knowledge_ablation/v2/direct_textbook_rag.jsonl \
  --output outputs/contracts/evidence_conditions.json
```

The full six-condition command is documented in [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md).

### 4. Prove the tool contract is learnable

```bash
python scripts/train_tool_sft.py \
  --config configs/knowledge/tool_sft_trace_no_knowledge.yaml \
  --limit 32 \
  --max-steps 100
```

Matched configurations include:

```text
configs/knowledge/tool_sft_trace_no_knowledge.yaml
configs/knowledge/tool_sft_irrelevant.yaml
configs/knowledge/tool_sft_textbook.yaml
configs/knowledge/tool_sft_anchors.yaml
configs/knowledge/tool_sft_combined.yaml
configs/knowledge/tool_sft_direct_textbook.yaml
```

A successful pilot requires valid tool syntax, non-empty assistant masks, zero truncation, falling loss, and improving held-out `finish_trace`/execution rates. A schema-only dry run is not evidence of learnability.

### 5. Optional on-policy refinement

```bash
python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml
```

Evidence-conditioned actors use `train_inverse_agent_knowledge.py`. The legacy independent-proof baseline uses `train_inverse_agent_trl.py`. Every required GRPO run loads a trainable Tool-SFT adapter and verifies its manifest, SHA-256, base-model revision, data contract, executor revision, and environment revision.

### 6. Generate frozen prediction artifacts

```bash
python scripts/infer_mechet.py \
  --config configs/agent/inverse_trace_grpo.yaml \
  --data data/benchmarks/h1/test.jsonl \
  --output outputs/h1/normal.jsonl \
  --mode trace \
  --condition-name trace_no_knowledge \
  --samples-per-target 4 \
  --seed 17 \
  --resume
```

The canonical runner streams `artifact_type=prediction` rows, supports resumable generation, and records model/tokenizer revisions, adapter hash, software versions, global seed, per-candidate seed, and selector version.

### 7. Evaluate without denominator leakage

```bash
python scripts/evaluate_faithfulness.py \
  --reference data/benchmarks/h1/test.jsonl \
  --normal outputs/h1/normal.jsonl \
  --intervention remove_tool_observations=outputs/h1/remove.jsonl \
  --intervention stale_tool_observations=outputs/h1/stale.jsonl \
  --intervention shuffle_tool_observations=outputs/h1/shuffle.jsonl \
  --output outputs/h1/summary.json

python scripts/evaluate_knowledge_ablation.py \
  --reference data/knowledge_ablation/v2/trace_textbook_rag.jsonl \
  --condition trace_no_knowledge=outputs/h3/trace.jsonl \
  --condition trace_textbook_rag=outputs/h3/textbook.jsonl \
  --condition direct_textbook_rag=outputs/h3/direct.jsonl \
  --output outputs/h3/summary.json
```

Prediction artifacts are distinct from supervision rows. Missing predictions remain in the denominator as failures; duplicate or extra IDs are hard errors.

## Evaluation semantics

Candidate rollouts are independent generations in generation order. Without a frozen ranking score, the correct metric is **Pass@K**, not Top-K.

Implemented metrics include:

```text
StructuralEndpointPass@1/5/10
MappedEndpointPass@1/5/10
ExecutePass@1/5/10
TraceBoundPass@1/5/10
coverage and selective risk
abstention rate
tool-failure recovery
retrieval Recall@K / Precision@K when frozen labels exist
retrieval latency
missing-prediction and re-execution error rates
```

Reaction-center and synthon metrics remain unavailable until frozen reference labels are supplied; they are not inferred from endpoints.

## Scope and non-claims

The current formal scope is mapped, closed-shell, two-electron polar organic chemistry supported by the deterministic executor.

The software alone does **not** establish:

- a unique physical mechanism from product alone;
- activation barriers, favorable kinetics, yield, or experimental success;
- universal condition compatibility;
- correctness for radical, photochemical, organometallic, spin, or coordination chemistry outside the declared scope;
- chemical truth from a citation, retrieval match, or learned score.

## Current status

| Layer | Status |
|---|---|
| Deterministic proof executor | Implemented and CI-tested |
| Trace-owned move replay and proof compilation | Implemented and CI-tested |
| Explicit TRL tool facades | Implemented and CI-tested |
| Root-import-preserving proof-to-trace conversion | Implemented and CI-tested |
| Replay-verified Tool-SFT construction | Implemented; full-data coverage not yet reported |
| Canonical seeded/resumable inference | Implemented and CI-tested |
| Strict H1/H3 evaluators | Implemented and CI-tested |
| H2 source-to-sink composition split | Implemented; benchmark statistics not yet released |
| Paper-scale checkpoints and frozen results | Not released |
| Experimental or kinetic validation | External evidence required |

No positive H1, H2, or H3 conclusion is claimed before full-data coverage, real model overfit, non-scripted inference, multi-seed training, and frozen benchmark evaluation.

## Documentation

| Document | Role |
|---|---|
| [`docs/SCIENTIFIC_THESIS.md`](docs/SCIENTIFIC_THESIS.md) | Scientific question, hypotheses, terminology, and permitted claims |
| [`docs/TRACE_FAITHFULNESS.md`](docs/TRACE_FAITHFULNESS.md) | Main causal runtime and H1 intervention contract |
| [`docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md`](docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md) | Paper-level claim–evidence contract |
| [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md) | Ordered commands, artifacts, gates, and stopping rules |
| [`docs/TOOL_SFT.md`](docs/TOOL_SFT.md) | Replay-verified supervision and checkpoint lineage |
| [`docs/PROOF_EQUIVALENCE.md`](docs/PROOF_EQUIVALENCE.md) | H2 execution-primitive signatures and composition splits |
| [`docs/KNOWLEDGE_ABLATIONS.md`](docs/KNOWLEDGE_ABLATIONS.md) | H3 matched evidence conditions and interventions |
| [`docs/README.md`](docs/README.md) | Documentation authority map and reading paths |
