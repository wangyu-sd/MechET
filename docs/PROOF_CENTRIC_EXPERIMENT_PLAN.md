# Legacy proof-centric implementation contract

> **Role:** detailed implementation and integrity gates. Paper-level experiment
> names and priorities are controlled by
> [`PAPER_EXPERIMENT_PROTOCOL.md`](PAPER_EXPERIMENT_PROTOCOL.md).
> **Commands:** [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md)  
> **Detailed runtime and data contracts:** [`TRACE_FAITHFULNESS.md`](TRACE_FAITHFULNESS.md), [`TOOL_SFT.md`](TOOL_SFT.md), [`PROOF_EQUIVALENCE.md`](PROOF_EQUIVALENCE.md), and [`KNOWLEDGE_ABLATIONS.md`](KNOWLEDGE_ABLATIONS.md)

## Research question

> Can mechanistic reasoning in retrosynthesis be made causal and compositional, rather than merely plausible in language?

## Experimental object

The unit of analysis is not a free-form rationale. It is a model-issued, environment-verified electron-flow program:

```text
mapped product
  -> explicit source-to-sink actions
  -> environment-owned state transitions
  -> immutable move trace
  -> finish_trace
  -> replay declared moves
  -> deterministic MECH_PROOF v1 compilation
  -> executor-derived endpoint views
```

The main environment exposes declared tools only. Internal state helpers and independent proof submission are not available to the model-facing main path.

## Claim matrix

| Claim | Primary comparison | Mandatory controls | Integrity gate | Falsifier |
|---|---|---|---|---|
| **H1: trace is causally used** | Normal trace-owned inference vs observation interventions | Removed, stale, shuffled observations; disabled state inspection/execution; direct and independent-proof baselines | Same model, adapter, revision, seed policy, K, and generation budget; explicit successful `finish_trace`; paired effects | Interventions produce no material behavioral change |
| **H2: known primitives compose out of distribution** | IID vs primitive-seen/composition-unseen test | Direct, CoT, net-edit, independent-proof, and trace-owned representations | `source_to_sink_execution_moves_v1`; all test primitives seen in train; zero complete-composition overlap | Test requires unseen primitives or is explained by near-duplicate structure/template overlap |
| **H3: external evidence adds information** | Textbook/anchor conditions vs trace-only and matched context controls | Length-matched irrelevant text, direct open-book, passage shuffle, same-topic wrong passage, warning/competitor removal | Frozen evidence, inference-available query, zero direct reward, matched runtime contract | Gain disappears against controls or depends on leakage/runtime mismatch |

## H1 — causal faithfulness

### Claim

The executable trace is a causal computational path to the precursor, not a post-hoc explanation.

### Required systems

```text
outcome-only generation
free-form CoT plus answer
state-CoT plus answer
net edit
independent complete proof
legacy loose trace plus submitted proof
trace-owned finish_trace method
```

### Required interventions

```text
remove_tool_observations
stale_tool_observations
shuffle_tool_observations
disable_inspect_state
disable_intermediate_execution
```

### Evaluation contract

A trace prediction is credited only when:

1. the model explicitly calls `finish_trace`;
2. the environment finalizes the episode;
3. the committed flow trace exists;
4. declared moves replay exactly;
5. trace, move-sequence, proof, and endpoint artifacts recompute without disagreement.

The evaluator does not complete an unfinished trace and does not parse a free-form answer in a trace condition.

### Claim gate

```text
same frozen ID universe
same model, adapter, model/tokenizer revision, seed policy, and generation budget
all missing predictions retained as failures
normal path trace-bound and re-executable
intervention construction audited
paired effect sizes reported
```

Endpoint accuracy without observation sensitivity is insufficient for H1.

## H2 — compositional basis

### Claim

Known local source-to-sink execution primitives generalize to unseen complete move compositions.

### Split contract

```text
primitive_basis = source_to_sink_execution_moves_v1
all test primitives observed in train at the declared minimum frequency
zero train/test complete-composition overlap
non-empty validation and test sets
split frozen before final model evaluation
```

Knowledge-anchor IDs and net `MECH_PROOF v1` bond/charge deltas are excluded from the headline split definition.

### Required comparisons

```text
outcome-only direct generation
free-form CoT
state-CoT
reaction-center or synthon prediction when frozen labels exist
net edit
independent complete proof
trace-owned source-to-sink Tool-CoT
```

### Required structural audits

Composition novelty must be separated from:

```text
exact product overlap
exact reaction overlap
product scaffold similarity
reaction-center template overlap
reaction family
ring formation or ring change
step count, move count, and proof topology
```

### Claim gate

A positive H2 result requires performance to be reported as a function of composition novelty and structural overlap. Primitive coverage alone is not evidence of broad chemical generalization; structural novelty alone is not evidence of primitive composition.

## H3 — evidence separation

### Claim

External mechanistic evidence improves program induction beyond trace ownership and additional context alone, while remaining subordinate to execution.

### Frozen conditions

| Condition | Endpoint path | Evidence intervention |
|---|---|---|
| `trace_no_knowledge` | Trace-owned | None |
| `trace_length_matched_irrelevant` | Trace-owned | Irrelevant text with the same character budget |
| `trace_textbook_rag` | Trace-owned | Frozen textbook evidence |
| `trace_structured_anchors` | Trace-owned | Frozen structured anchors |
| `trace_text_plus_anchors` | Trace-owned | Textbook plus anchors |
| `direct_textbook_rag` | Direct answer | The same bounded textbook evidence |

### Evidence controls

```text
passage_shuffle
same_topic_wrong
remove_warnings
remove_competing_pathways
same bounded evidence for direct and trace comparisons
zero direct evidence reward
```

### Claim gates

A textbook claim requires:

```text
trace_textbook_rag > trace_no_knowledge
and
trace_textbook_rag > trace_length_matched_irrelevant
```

A combined-evidence claim requires:

```text
trace_text_plus_anchors > trace_textbook_rag
and
trace_text_plus_anchors > trace_structured_anchors
```

These comparisons require the same base-model revision and generation contract. Condition-specific adapter hashes and supervised-token-normalized compute are reported rather than hidden.

## Data contract

Every example has a stable ID and separates:

```text
full_precursor_state
structural_precursor
auxiliary_fragments
```

The primary endpoint is atom-contributing `structural_precursor` exact match with atom-map labels ignored. Mapped exact match is secondary.

Proof-to-trace conversion must:

- preserve root and edge imports;
- reject ambiguous electron pairing;
- use inference-available query information in headline conditions;
- replay the exact terminal endpoint;
- report family and complexity coverage;
- emit stable quarantine reasons, including tool-budget overflow.

## Tool-SFT contract

Every trace-owned training row contains:

```text
messages
tools
JSON-object tool arguments
one result per tool call
exactly one successful finish_trace
trace and move-sequence digests
executor replay metadata
frozen endpoint views
```

Real training must pass:

```text
frozen model and tokenizer revision
real chat-template rendering
non-empty assistant masks
zero headline truncation
frozen data hash
adapter manifest and SHA-256
fixed training seed and data seed
```

Paper-scale on-policy training begins only after a real small-set overfit demonstrates learnable tool syntax and improved held-out completion/execution rates.

## Prediction artifact contract

Every model output uses `artifact_type=prediction` and records:

```text
condition and prediction mode
complete messages and tools
candidate rollouts and candidate seeds
rollout_state and raw terminal result
base model, tokenizer, revision, and adapter hash
global seed and selector version
temperature, top-p, token, iteration, and candidate budgets
software versions and intervention metadata
```

Evaluation uses one frozen reference universe. Duplicate or extra IDs and supervision rows are hard errors. Missing predictions remain in the denominator as failures.

## Metric contract

Independent candidate generations are reported as **Pass@K** unless a frozen, gold-independent ranking score is stored.

Implemented metrics include:

```text
StructuralEndpointPass@1/5/10
MappedEndpointPass@1/5/10
ExecutePass@1/5/10
TraceBoundPass@1/5/10
coverage and selective risk
abstention rate
tool-failure recovery
retrieval Recall@K and Precision@K with frozen labels
retrieval latency
missing-prediction and re-execution error rates
paired intervention effects
```

Reaction-center and synthon metrics remain unavailable until frozen labels exist.

## Formal and empirical evidence

The deterministic executor is the hard source of formal validity. Textbook passages, structured anchors, and an independent forward expert are soft evidence. No learned score can rescue an invalid trace or proof.

A forward-evidence result additionally requires:

```text
independently frozen and calibrated model
held-out or cross-fitted lineage
explicit competitor sets for selectivity
family-wise false-acceptance and false-rejection analysis
```

## Scale and planning

Scale studies, RL reward decomposition, textbook/RAG, K-hypothesis search, and
multistep planning are downstream or separate studies. They begin only after
the paper's matched SFT, R3 and R4-C2 results are frozen and cannot rescue a
failed causal or compositional claim.

## Prohibited interpretations

Current software alone cannot establish:

```text
unique physical mechanism from product alone
activation barriers or kinetic preference
yield or laboratory success
universal condition compatibility
radical, photochemical, organometallic, spin, or coordination chemistry outside scope
reaction discovery without external validation
```

## Reproducibility package

Every reported checkpoint and result must include:

```text
repository and data revisions
source licenses and hashes
split, coverage, and quarantine manifests
model and tokenizer revisions
adapter manifests and hashes
training configs, seeds, and optimizer updates
GPU type, wall time, and compute disclosure
tokenizer input and supervised-token counts
prediction manifests
evaluation outputs and claim-gate status
```
