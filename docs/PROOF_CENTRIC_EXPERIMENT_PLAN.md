# MechET authoritative scientific experiment contract

This document defines the frozen paper claims and evidence required for the causal and compositional MechET study. Operational commands are in `EXECUTION_PLAN.md`; detailed contracts are in `TRACE_FAITHFULNESS.md`, `TOOL_SFT.md`, `PROOF_EQUIVALENCE.md`, and `KNOWLEDGE_ABLATIONS.md`.

## Central question

> Can mechanistic reasoning in retrosynthesis be made causal and compositional, rather than merely plausible in language?

## Main method

```text
mapped product
  -> explicit source-to-sink actions
  -> environment-owned state transitions
  -> immutable move trace
  -> replay declared moves
  -> deterministic MECH_PROOF v1 compilation
  -> executor-derived full precursor state
  -> atom-contributing structural precursor
```

The model-facing main environment is an explicit TRL facade. It exposes only the declared tools and never exposes internal state helpers or independent proof submission.

## H1 — causal faithfulness

### Claim

The model's executable trace is a causal computational path to the precursor, not a post-hoc explanation.

### Required baselines

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
remove tool observations
stale tool observations
shuffle tool observations
disable inspect_state
disable intermediate execution
```

### Integrity gates

```text
same frozen ID universe
same model, adapter, revision, K, and generation budget
missing predictions count as failures
normal path replays declared moves and proof
paired effects reported
```

Insensitivity to observations blocks H1 regardless of endpoint accuracy.

## H2 — compositional basis

### Claim

Known source-to-sink execution primitives generalize to unseen complete move compositions.

### Split contract

```text
primitive_basis = source_to_sink_execution_moves_v1
all test primitives seen in train
zero train/test composition overlap
non-empty held-out test
fixed minimum train primitive frequency
```

The split is built from replay-verified trace plans. Knowledge-anchor IDs and MECH_PROOF net deltas are excluded from the headline definition.

### Required reporting

```text
IID and composition-OOD
composition frequency and novelty
steps and move counts
family/scaffold/ring/topology strata
primitive coverage and split quarantine
```

## H3 — evidence separation

### Claim

External mechanistic evidence improves selection or induction beyond trace ownership and additional context alone, while remaining subordinate to execution.

### Frozen conditions

```text
trace_no_knowledge
trace_length_matched_irrelevant
trace_textbook_rag
trace_structured_anchors
trace_text_plus_anchors
direct_textbook_rag
```

### Evidence controls

```text
passage shuffle
same-topic wrong passage
remove warnings
remove competing pathways
same bounded evidence for direct and trace conditions
zero direct evidence reward
```

### Claim gates

Textbook:

```text
textbook > trace-only
and
textbook > length-matched irrelevant
```

Combined:

```text
combined > textbook
and
combined > anchors
```

All prediction artifacts use the same base/revision and generation budget; condition-specific adapter hashes and token-normalized compute are reported.

## Data contract

Every example has a stable ID and distinguishes:

```text
full_precursor_state
structural_precursor
auxiliary_fragments
```

The primary endpoint is atom-contributing structural precursor exact match with atom maps ignored. Mapped exact is secondary.

Proof-to-trace conversion preserves root and edge imports, rejects ambiguous electron pairing, uses only inference-available query information in headline conditions, and emits family/complexity coverage and stable quarantine reasons.

## Tool-SFT contract

Each trace row contains `messages`, a canonical `tools` schema, JSON-object arguments, matched tool-call/result pairs, exactly one `finish_trace`, replay metadata, and trace/move digests.

Real training must pass:

```text
real tokenizer rendering
non-empty assistant mask
zero truncation
valid tool schemas
frozen data hash
adapter manifest/hash
```

GRPO loads the corresponding Tool-SFT adapter as trainable PEFT state and validates its base model, hash, data contract, and executor/environment revisions.

## Prediction artifact contract

Every model output is `artifact_type=prediction` and records:

```text
condition and prediction mode
complete messages and tools
candidate rollouts
rollout_state and terminal result
base model and adapter hash
model revision
temperature, top-p, token, iteration, and K budgets
intervention metadata
```

Evaluation uses one frozen reference universe. Duplicate/extra IDs and supervision rows are hard errors; missing predictions remain failures.

## Metrics

Implemented:

```text
structural and mapped Top-1/5/10
ExecutePass@1/5/10
TraceBoundPass@1/5/10
coverage and selective risk
abstention
tool-failure recovery
retrieval recall/precision with gold passage labels
retrieval latency
missing and re-execution error rates
paired intervention effects
```

Reaction-center and synthon metrics remain null until frozen labels exist.

## Formal and empirical evidence

The deterministic executor is the hard source of formal validity. Text, anchors, and the independent forward expert are soft evidence. No learned score can rescue an invalid trace or proof.

A forward-evidence result requires an independently frozen/calibrated model, explicit competitor sets for selectivity, cross-fitting or held-out lineage, and family-wise false-acceptance/false-rejection reporting.

## Scale and planning

Scale, on-policy reward decomposition, K-hypothesis search, and multistep planning are downstream results. They begin only after H1–H3 pilots pass and cannot rescue a failed causal or compositional claim.

## Prohibited claims

Current software alone cannot establish:

```text
unique physical mechanism from product alone
activation barriers or kinetics
yield or laboratory success
universal condition compatibility
radical, photochemical, organometallic, spin, or coordination chemistry outside scope
reaction discovery without external validation
```

## Reproducibility package

Each reported checkpoint/result must include:

```text
repository and data revisions
source licenses and hashes
split and quarantine manifests
model/tokenizer revisions
adapter manifests and hashes
training configs and seeds
optimizer updates and GPU hours
tokenizer input/supervised token counts
prediction manifests
evaluation outputs and claim gates
```
