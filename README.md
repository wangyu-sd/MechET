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

> **Paper authority:** the ICLR experiment definitions and priorities follow
> [`MechET-paper/EXPERIMENT_MATRIX.md`](https://github.com/wangyu-sd/MechET-paper/blob/main/EXPERIMENT_MATRIX.md).
> The implementation mapping is frozen in
> [`docs/PAPER_EXPERIMENT_PROTOCOL.md`](docs/PAPER_EXPERIMENT_PROTOCOL.md) and
> [`configs/experiments/paper_experiment_matrix_v1.yaml`](configs/experiments/paper_experiment_matrix_v1.yaml).

## Project status

> **Dataset-lineage warning:** before describing a FlowER split as "full" or
> submitting a FlowER job, read [`PROJECT_MEMORY.md`](PROJECT_MEMORY.md). The
> full reaction-level split is **257,171 / 2,890 / 28,971**; the 32k proof and
> 28k/27k executable-trace artifacts are legacy derived subsets. The separate
> v4 full-reaction artifact retains all rows and labels its eight corrupt
> upstream endpoint fallbacks explicitly.

> [!CAUTION]
> **Incomplete trace-view subsets:** FlowER `3,080` and mech-USPTO-31k
> `1,124` are replay-compatible **test subsets**, not complete benchmark test
> sets. They are diagnostic/program-analysis views only and must never be
> reported as full-test or headline results. The complete denominators are
> FlowER `28,971` and mech-USPTO-31k `3,120`.

> [!WARNING]
> **mech-USPTO mapping source:** `rxn_prod_min` is unmapped and must never be
> copied into a field named `product_mapped`. The active public-source protocol
> reconstructs complete reaction pairs from HF `elem_reac_spe` step 0 and the
> principal `rxn_prod_min` product, maps them once with pinned RXNMapper, then
> applies synchronized product-only canonical reindexing. See
> [`docs/MECH_USPTO_31K_FULL_ENDPOINT.md`](docs/MECH_USPTO_31K_FULL_ENDPOINT.md).

| Layer | Current status |
|---|---|
| Deterministic executor and trace-owned runtime | Implemented and CI-tested |
| Replay-verified Tool-SFT data contract | Implemented; real Qwen3-8B one-step QLoRA runtime smoke passed |
| FlowER full-reaction trace v4 | 257,171 / 2,890 / 28,971 rows; 289,024 strict traces + 8 labelled endpoint fallbacks; zero split-ID overlap |
| FlowER A7 action-only screen | `action_delta_v1` frozen at 257,167 / 2,890 / 28,967 strict executable rows with zero state leaks; frozen cache is one lossless window per train row; real 8×A100 gradient training is active and the unused H20 candidate was stopped |
| Full FlowER P0 control screens | A0 Direct and A2 State-CoT: 257,171 train rows; A3 NetEdit and auxiliary complete-proof: 257,167 strict-proof rows; all four 3-epoch adapters exist; clean complete-proof K=10 inference is running and A0/A2/A3 are queued with EOS-safe batched decoding; final Tier-A still requires common IDs and matched supervised-token budget |
| mech-USPTO-31k inverse Tool-SFT v2 | 11,429/11,429 stitched traces accepted; replay/tokenizer audits passed; action-only Qwen3-8B completed three epochs on the 9,118-row train subset and its explicitly non-headline 1,124-row K=10 program-view rollout is queued |
| Qwen3 assistant-only token contract | Implemented; final-sequence mask and zero-truncation audit |
| Future textbook/RAG evidence isolation | Implemented infrastructure; outside current ICLR protocol |
| H1 causal-faithfulness result | **Not established** |
| H2 compositional-generalization result | **Not established** |
| Textbook/RAG/H3 | Outside the current ICLR paper protocol |
| Paper-scale benchmark and checkpoints | Pending |

See [`docs/STATUS_MATRIX.md`](docs/STATUS_MATRIX.md) for the complete implementation/evidence matrix. A working infrastructure component is not a positive scientific result.

## Research thesis

Most mechanism-aware language models optimize an answer and a rationale jointly, but do not guarantee that the rationale is causally upstream of the answer. A chemically plausible chain of thought may therefore coexist with an endpoint produced through an independent, unverifiable channel.

MechET removes that ambiguity by formulating retrosynthesis as **causal program induction over explicit source-to-sink electron-flow actions**. The model commits actions to a deterministic environment; an environment-owned trace is then the only computational source of the executable proof and precursor.

The ICLR study is organized around a seven-part evidence chain (R1--R7), with
two central mechanistic tests: counterfactual state adaptation and MechComp-OOD
C2. The older H1/H2 names remain useful shorthand for these two tests:

| Hypothesis | Scientific question | Required evidence | Falsifying outcome |
|---|---|---|---|
| **H1 — causal faithfulness** | Does the committed tool trace determine the endpoint? | Successful `finish_trace`, trace/proof replay, and paired observation interventions under an identical runtime contract | Endpoint performance is unchanged when chemically relevant tool observations are removed, stale, or shuffled |
| **H2 — compositional basis** | Can familiar local execution primitives form unseen complete mechanisms? | Primitive-seen/composition-unseen splits built from `source_to_sink_execution_moves_v1` | Held-out examples contain unseen primitives, or performance is explained only by scaffold/template overlap |
| **H3 — evidence separation** *(future study)* | Does external textbook evidence add information beyond the executable trace? | Frozen retrieval, length-matched irrelevant text, passage shuffle and content controls | Any gain disappears under matched controls or depends on label-oracle information |
| **R1/R2 — performance and provenance** | Does the executable representation recover precursors while exposing valid, endpoint-consistent programs? | Matched A0--A7 controls across FlowER and mech-USPTO | Gains vanish under matched data/compute or reasoning is incompatible with the endpoint |

The complete paper sequence additionally covers data efficiency, structural
OOD, cross-corpus transfer, theory-linked analyses and recovery. Textbook/RAG
evidence experiments are retained in the repository as a separate future
study; they are not part of the current ICLR claim set.

The authoritative claim definitions and boundaries are in [`docs/SCIENTIFIC_THESIS.md`](docs/SCIENTIFIC_THESIS.md).

## Main method

```text
atom-mapped product
  -> use product atom maps and import missing mapped fragments
  -> explicit source-to-sink electron-flow actions
  -> environment-owned molecular-state transitions (executor-internal)
  -> action success/failure feedback (model-visible default)
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

### Observation contract

The main A7 representation is `action_delta_v1`: the product is supplied once,
but intermediate molecular-state SMILES are not placed in the model context.
The executor still keeps the complete mapped state, validates every action, and
uses that private state to compile and execute the terminal proof. `inspect_state`
returns the legal source/sink inventory without returning `state_smiles` in this
mode.

State visibility is an ablation, not a requirement of the main method:

| Builder mode | Model-visible transition result | Paper role |
|---|---|---|
| `action_delta` | success/failure, stable code and remaining budget | **Main/default** |
| `reaction_center_delta` | action result plus mapped one-hop changed neighbourhood | Representation ablation |
| `full_state` | complete legacy state after each tool call | Legacy/upper-bound ablation |

All three modes execute the identical actions with the same private executor.
They must be compared on identical stable IDs and action targets. Report endpoint
and execution metrics together with invalid-action rate, recovery, context tokens,
training throughput and rollout latency. The existing
`data/flower_inverse_tool_sft_full_v4` artifact predates this contract and contains
full-state observations; it must not be labelled as the action-only main result.

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

The current ICLR paper logic is deliberately sequential:

```text
coverage and leakage audit
  -> R1/R2 matched performance and provenance
  -> R3 counterfactual state adaptation
  -> R4 MechComp-OOD, with C2 as headline
  -> R5 data efficiency
  -> R6 OOD and cross-corpus transfer
  -> R7 theory-linked analysis and recovery
  -> optional RL, scale, textbook/RAG, and planning
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

### Future study — textbook/RAG evidence separation

This implemented suite is not part of the current ICLR experiment matrix. It
is retained for a later evidence-conditioning study.

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

### FlowER data prerequisite

The public FlowER archive provides `flower_new_dataset` elementary
trajectories.  The reaction-level `flower_retro` files required by the full
endpoint track are derived locally; they are not an undocumented second
download:

```bash
# Download and unzip Figshare data.zip as documented in data/README.md, then:
export FLOWER_ROOT=data/raw/data/flower_new_dataset

python scripts/build_flower_retro.py \
  --flower-root "$FLOWER_ROOT" \
  --output-dir "$(dirname "$FLOWER_ROOT")/flower_retro" \
  --splits train valid test

python scripts/build_flower_full_endpoint_sft.py \
  --data-root "$(dirname "$FLOWER_ROOT")" \
  --output-dir data/flower_full_endpoint_sft \
  --splits train valid test
```

The derivation, canonical row counts, and SHA-256 values are documented in
[`data/README.md`](data/README.md#derive-flower_retro-no-separate-download).

### 2. Build replay-verified trajectories for all splits

For the separate mech-USPTO-31k inverse initialization dataset, use the frozen
download-to-training pipeline in
[`docs/MECH_USPTO_31K_INVERSE_TOOL_SFT.md`](docs/MECH_USPTO_31K_INVERSE_TOOL_SFT.md).
It contains 9,118 train, 1,187 valid, and **1,124 incomplete trace-view test
subset** rows. This
`trace_no_knowledge` dataset does not use the textbook corpus and is distinct
from FlowER and the USPTO-50K benchmark.

```bash
for split in train valid test; do
  python scripts/build_textbook_tool_sft.py \
    --input data/mechet_proof_clean/${split}.jsonl \
    --corpus knowledge/corpus/passages.jsonl \
    --output data/textbook_tool_sft/${split}.jsonl \
    --observation-mode action_delta \
    --query-mode state

  python scripts/build_textbook_tool_sft.py \
    --input data/mechet_proof_clean/${split}.jsonl \
    --corpus knowledge/corpus/passages.jsonl \
    --output data/textbook_tool_sft/${split}_text_and_anchors.jsonl \
    --enable-structured-primitives \
    --observation-mode action_delta \
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

### Optional future study: run held-out H3

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
| Replay-verified Tool-SFT construction | Implemented; mech-USPTO inverse v2 accepts 11,429/11,429 globally stitched traces |
| Qwen3 final-sequence assistant masking | Implemented; mech-USPTO inverse full-data audit has zero truncation |
| Split-isolated H3 suite construction | Implemented future-study infrastructure; outside current ICLR |
| Canonical seeded/resumable inference | Implemented and CI-tested |
| Strict intervention/evidence evaluators | Implemented and CI-tested |
| R3/R4 reproducible runners | Implemented foundations; paper checkpoints and splits still required |
| H2 source-to-sink composition split | Implemented; benchmark statistics not yet released |
| Paper-scale checkpoints and frozen results | Not released |
| Experimental or kinetic validation | External evidence required |

No positive R1--R7 conclusion is claimed before its paper-declared matched
data, compute, inference, multi-seed and frozen-evaluation contracts pass.

## Documentation

| Document | Role |
|---|---|
| [`docs/PAPER_EXPERIMENT_PROTOCOL.md`](docs/PAPER_EXPERIMENT_PROTOCOL.md) | Paper-authoritative A0--A7, B1--B5 and R1--R7 implementation mapping |
| [`docs/SCIENTIFIC_THESIS.md`](docs/SCIENTIFIC_THESIS.md) | Runtime terminology and permitted claim boundaries |
| [`docs/TRACE_FAITHFULNESS.md`](docs/TRACE_FAITHFULNESS.md) | Main causal runtime and H1 intervention contract |
| [`docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md`](docs/PROOF_CENTRIC_EXPERIMENT_PLAN.md) | Legacy proof-centric implementation contract |
| [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md) | Ordered commands, artifacts, gates, and stopping rules |
| [`docs/TOOL_SFT.md`](docs/TOOL_SFT.md) | Replay-verified supervision, Qwen3 assistant masking, and checkpoint lineage |
| [`docs/MECH_USPTO_31K_INVERSE_TOOL_SFT.md`](docs/MECH_USPTO_31K_INVERSE_TOOL_SFT.md) | mech-USPTO source identity, inverse v2 protocol, coverage, validation, and reproduction commands |
| [`docs/PROOF_EQUIVALENCE.md`](docs/PROOF_EQUIVALENCE.md) | H2 execution-primitive signatures and composition splits |
| [`docs/KNOWLEDGE_ABLATIONS.md`](docs/KNOWLEDGE_ABLATIONS.md) | Future textbook/RAG evidence conditions and interventions |
| [`docs/README.md`](docs/README.md) | Documentation authority map and reading paths |
