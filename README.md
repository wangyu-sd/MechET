<div align="center">

<img src="docs/assets/mechet-icon.png" alt="MechET project icon" width="168" />

# MechET

**Causal and compositional electron-flow reasoning for retrosynthesis**

[![Proof tests](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/proof-tests.yml)
[![Runtime tests](https://github.com/wangyu-sd/MechET/actions/workflows/agent-framework-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/agent-framework-tests.yml)
[![Evidence tests](https://github.com/wangyu-sd/MechET/actions/workflows/knowledge-ablation-tests.yml/badge.svg)](https://github.com/wangyu-sd/MechET/actions/workflows/knowledge-ablation-tests.yml)

[Research thesis](#research-thesis) · [Main method](#main-method) · [Experimental program](#experimental-program) · [Reproduce](#minimum-credible-reproduction) · [Documentation](#documentation)

</div>

---

> **MechET asks whether a mechanistic rationale can be the computation that determines a retrosynthetic prediction—not merely a plausible explanation written after the answer.**

## Project status

| Layer | Current status |
|---|---|
| Deterministic executor and trace-owned runtime | Implemented and CI-tested |
| Replay-verified Tool-SFT data contract | Implemented; real-model overfit pending |
| Qwen3 assistant-only token contract | Implemented; final-sequence mask and zero-truncation audit |
| Train/valid/test evidence isolation | Implemented; final H3 uses held-out `test/` only |
| H1 causal-faithfulness result | **Not established** |
| H2 compositional-generalization result | **Not established** |
| H3 evidence-benefit result | **Not established** |
| Paper-scale benchmark and checkpoints | Pending |

See [`docs/STATUS_MATRIX.md`](docs/STATUS_MATRIX.md) for the complete implementation/evidence matrix. A working infrastructure component is not a positive scientific result.

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

Scaffold, family, step-state reaction-center, and near-duplicate overlap must be reported separately rather than conflated with composition novelty. The H2 model is retrained **after** the composition split on H2/train only.

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

The main retrieval query uses only inference-available molecular-state terms. `label_oracle` is an explicitly named upper bound and is not headline-eligible. Conditions are constructed independently inside `train/`, `valid/`, and `test/`; final H3 results use the frozen `test/` universe only.

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

### 2. Build replay-verified trajectories for all splits

```bash
for split in train valid test; do
  python scripts/build_textbook_tool_sft.py \
    --input data/mechet_proof_clean/${split}.jsonl \
    --corpus knowledge/corpus/passages.jsonl \
    --output data/textbook_tool_sft/${split}.jsonl \
    --query-mode state

  python scripts/build_textbook_tool_sft.py \
    --input data/mechet_proof_clean/${split}.jsonl \
    --corpus knowledge/corpus/passages.jsonl \
    --output data/textbook_tool_sft/${split}_text_and_anchors.jsonl \
    --enable-structured-primitives \
    --query-mode state
done
```

Rows exceeding the frozen headline tool budget are quarantined rather than trained under a larger hidden budget.

### 3. Build and validate split-isolated matched conditions

```bash
python scripts/build_knowledge_ablation_suite.py \
  --config configs/experiments/textbook_ablation.yaml

python scripts/validate_experiment_contract.py \
  --model-name Qwen/Qwen3-0.6B \
  --condition trace_no_knowledge=data/knowledge_ablation/v2/train/trace_no_knowledge.jsonl \
  --condition trace_textbook_rag=data/knowledge_ablation/v2/train/trace_textbook_rag.jsonl \
  --condition direct_textbook_rag=data/knowledge_ablation/v2/train/direct_textbook_rag.jsonl \
  --output outputs/contracts/evidence_conditions_train.json
```

The suite builder requires disjoint train/valid/test stable IDs. The full six-condition train/valid/test validation commands are documented in [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md).

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

Qwen3 assistant masks are produced by scanning the final tokenized ChatML sequence, not by `return_assistant_tokens_mask=True`. All six conditions use `max_length=12288`; headline rows are forbidden to truncate. A mutable request such as `model_revision: main` is resolved and recorded as the actual immutable 40-hex model/tokenizer commit.

A successful pilot requires valid tool syntax, non-empty assistant masks, zero truncation, falling loss, and improving held-out `finish_trace`/execution rates. A schema-only dry run is not evidence of learnability.

### 5. Optional on-policy refinement

Portable/default:

```bash
python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo.yaml
```

Paper-scale vLLM profile:

```bash
python scripts/train_inverse_agent_trace.py \
  --config configs/agent/inverse_trace_grpo_vllm.yaml
```

Evidence-conditioned actors use `train_inverse_agent_knowledge.py`. The legacy independent-proof baseline uses `train_inverse_agent_trl.py`. Every required GRPO run loads a trainable Tool-SFT adapter and verifies its manifest, SHA-256, immutable base-model revision, data contract, executor revision, and environment revision.

### 6. Freeze and run H1

```bash
python scripts/build_h1_benchmark.py \
  --input data/knowledge_ablation/v2/test/trace_no_knowledge.jsonl \
  --train-reference data/knowledge_ablation/v2/train/trace_no_knowledge.jsonl \
  --output-dir data/benchmarks/h1

python scripts/run_h1_suite.py \
  --data data/benchmarks/h1/test.jsonl \
  --out-dir outputs/h1 \
  --samples-per-target 4 \
  --seed 17 \
  --resume
```

The canonical runner uses `scripts/infer_mechet.py`, streams `artifact_type=prediction` rows, and records model/tokenizer revisions, adapter hash, software versions, global seed, per-candidate seed, and selector version. It evaluates with `scripts/evaluate_faithfulness.py`.

### 7. Build, retrain, and run H2

```bash
python scripts/build_mechcomp_ood.py \
  --input data/knowledge_ablation/v2/train/trace_no_knowledge.jsonl \
  --output-dir data/ood/mechcomp_source_sink \
  --min-train-primitive-count 5 \
  --seed 42

python scripts/train_tool_sft.py \
  --config configs/agent/tool_sft_mechcomp_trace.yaml

python scripts/run_h2_suite.py \
  --split-dir data/ood/mechcomp_source_sink \
  --adapter outputs/h2/tool_sft_trace_qwen3_0_6b \
  --out-dir outputs/h2 \
  --samples-per-target 4 \
  --seed 17
```

`run_h2_suite.py` refuses an adapter whose `train_file_sha256` does not match the frozen H2/train split. The headline trace-owned path uses `scripts/infer_mechet.py --mode trace`; `scripts/infer_mechet_proof.py` is an independent complete-proof baseline only.

### 8. Run held-out H3

```bash
python scripts/run_h3_suite.py \
  --suite-root data/knowledge_ablation/v2/test \
  --out-dir outputs/h3 \
  --samples-per-target 4 \
  --seed 17
```

The runner checks that each condition-specific adapter was trained on the corresponding `data/knowledge_ablation/v2/train/` file before calling `scripts/evaluate_knowledge_ablation.py`.

For evidence-content controls such as `same_topic_wrong`, `scripts/build_evidence_interventions.py` writes a condition-specific `*.eligible_ids.json` and `*.reference.jsonl`. Any reduced intervention set must be evaluated against that paired reference, never against the full baseline ID universe.

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
| Qwen3 final-sequence assistant masking | Implemented; real full-data tokenizer audit pending |
| Split-isolated H3 suite construction | Implemented; final results not yet reported |
| Canonical seeded/resumable inference | Implemented and CI-tested |
| Strict H1/H3 evaluators | Implemented and CI-tested |
| H1/H2/H3 reproducible runners | Implemented; real checkpoints required |
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
| [`docs/TOOL_SFT.md`](docs/TOOL_SFT.md) | Replay-verified supervision, Qwen3 assistant masking, and checkpoint lineage |
| [`docs/PROOF_EQUIVALENCE.md`](docs/PROOF_EQUIVALENCE.md) | H2 execution-primitive signatures and composition splits |
| [`docs/KNOWLEDGE_ABLATIONS.md`](docs/KNOWLEDGE_ABLATIONS.md) | H3 matched evidence conditions and interventions |
| [`docs/README.md`](docs/README.md) | Documentation authority map and reading paths |
