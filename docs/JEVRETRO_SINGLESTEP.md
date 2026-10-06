# Reverse electron-transfer supervision for retrosynthesis

## Scientific question

The target claim is not that a retrosynthesis model should autoregressively
roll out a mechanism at inference time. The target claim is narrower:

> Does reverse electron-transfer supervision improve precursor prediction, and
> does that improvement translate into more effective multi-step planning?

The implementation therefore separates **training supervision** from
**inference behavior**. Reverse-ET appears only in training for the primary
causal experiment. At test time every condition receives the same product-only
prompt and predicts the same precursor representation.

## Primary matched experiment on FlowER

Use the frozen strict-program FlowER universe because it provides both endpoint
targets and reverse electron-transfer traces on almost the complete reaction
set:

- train: 257,167 reactions
- validation: 2,890 reactions
- strict test view: 28,967 reactions

All three conditions use exactly these reaction IDs.

### Condition A: endpoint only

For each reaction, train on product -> precursor. To match the number of
training examples per reaction, the endpoint example is presented twice.

### Condition B: endpoint + NetEdit

For each reaction, train on:

1. product -> precursor
2. product -> net bond/charge/hydrogen edit description

The NetEdit auxiliary target contains no precursor answer.

### Condition C: endpoint + Reverse-ET

For each reaction, train on:

1. product -> precursor
2. product -> complete reverse electron-transfer supervision program

The reverse-ET target contains fragment imports and the source-to-sink moves
for each recorded reverse step, but contains no precursor answer and is
generated in one shot. There is no executor rollout during endpoint inference.

The builder is:

    python scripts/build_reverse_et_transfer_supervision.py \
      --source-dir data/flower_inverse_tool_sft_action_delta_v1 \
      --output-dir data/reverse_et_transfer_v1

It creates matched training files:

    train_endpoint_only_matched.jsonl
    train_endpoint_plus_netedit.jsonl
    train_endpoint_plus_reverse_et.jsonl

Each contains exactly 2 x 257,167 examples and exactly the same underlying
reaction IDs. Validation uses endpoint prediction only for all three models.

## Why NetEdit is mandatory

Endpoint-only versus Reverse-ET cannot establish that electron transfer itself
is useful, because any extra transformation supervision may help. NetEdit is
therefore the main control.

The key contrast is:

    Reverse-ET - NetEdit

Both provide information about the transformation. Reverse-ET additionally
specifies electron source, electron sink and the grouping of coordinated
moves. A positive result requires Reverse-ET to improve beyond NetEdit under
the same endpoint task.

## Model and inference contract

First experiment:

- backbone: Qwen3-0.6B
- LoRA rank: 16
- same model revision
- same optimizer and seed
- same reaction IDs
- same endpoint validation set
- same endpoint inference prompt
- same K=10 decoding budget
- no mechanism generation at endpoint test time

The primary run is example-matched. If Reverse-ET is positive, a second
confirmatory run must match supervised-token budget using the repository's
existing token audit machinery. GPU-hours and actual supervised tokens are
reported for every condition.

The endpoint test output is always:

    product -> ranked precursor candidates

Never:

    product -> electron steps -> precursor

Thus the experiment measures whether reverse-ET supervision changes what the
model learns, not whether a long mechanism rollout can survive its own errors.

## Single-step evidence

Primary single-step metrics:

- structural Top-1 / Top-5 / Top-10
- mapped exact as secondary
- invalid candidate rate
- candidate diversity
- calibration / NLL of the reference precursor
- inference latency (identical endpoint inference contract)

The first causal comparison is on FlowER, where all three supervision signals
can be constructed on the same reactions.

A later transfer experiment uses USPTO-50K only as a standard endpoint-only
downstream benchmark:

1. initialize from Endpoint-only / NetEdit / Reverse-ET FlowER checkpoints;
2. fine-tune all three on exactly the same USPTO-50K endpoint train split;
3. use identical fine-tuning budget;
4. evaluate the standard 5,007-product test set.

USPTO-50K is not treated as mechanism data.

## Evidence for why Reverse-ET helps

The explanation must be empirical rather than rhetorical. Four analyses are
frozen in advance.

### 1. Reverse-ET versus NetEdit

If Reverse-ET does not outperform NetEdit, do not claim that electron
direction carries useful information beyond reaction-center/edit supervision.

### 2. Electron-transfer complexity

Report the endpoint gain of Reverse-ET over NetEdit as a function of:

- number of reverse electron-transfer steps;
- number of coupled moves per step;
- total number of electron moves;
- number of imported fragments.

This tests whether the extra supervision is useful specifically when the
reaction cannot be summarized by one simple bond edit.

### 3. Composition novelty

Reuse the existing MechComp/H2 signatures. Stratify endpoint prediction by
frequency of the complete move composition while keeping constituent
source-to-sink primitives seen in training.

The question is whether Reverse-ET supervision improves precursor prediction
when the complete transformation pattern is less familiar.

### 4. Disagreement audit

On reactions where Endpoint-only, NetEdit and Reverse-ET select different
Top-1 precursors, report:

- whether the reference precursor is recovered;
- reaction center overlap;
- net-edit agreement;
- reverse-ET program compatibility where an executable reference is available.

This identifies whether gains actually come from better transformation
selection rather than formatting or ranking artifacts.

## Multi-step planning experiment

Multi-step planning is a transfer test of the same three representations, not
a continuation of one reaction's electron trajectory.

Use PaRoutes n1 and n5. PaRoutes supplies 10,000 targets for each set, matching
stocks, reference routes and a USPTO-derived reaction dataset for training a
one-step model.

Protocol:

1. start from the three FlowER checkpoints;
2. fine-tune all three on the same PaRoutes one-step reaction training set
   using endpoint-only supervision;
3. freeze the three one-step models;
4. use the same Top-K precursor decoding;
5. plug each model into the existing Syntheseus BackwardReactionModel adapter;
6. run the same Retro* implementation, stock and search budgets.

Reverse-ET is not rolled out inside the planner. Each planner edge is simply a
single-step precursor prediction made by the corresponding endpoint model.

Primary planning metrics:

- solved targets;
- solved rate versus reaction-model-call budget;
- first-solution model calls;
- first-solution wall time;
- reference-route Top-1 / Top-5 / Top-10;
- route length and number of expanded nodes.

PaRoutes provides two 10,000-route benchmark sets and reports solved-target and
reference-route Top-N metrics, making it suitable for the multi-step comparison.

## Mechanism-to-planning explanation

The planning explanation is tested through the one-step model, not assumed.

For every reference route edge in PaRoutes, record the rank assigned by the
three one-step models to the reference disconnection. Then test whether:

1. Reverse-ET improves reference-edge rank relative to NetEdit;
2. targets with improved edge ranks require fewer Retro* model calls;
3. the solved-rate improvement disappears when the planner is forced to use
   the same edge ranking.

This links any planning gain to better one-step reaction ordering instead of
changes in the search algorithm.

## What PR #81 remains useful for

PR #81 answers a different question: whether explicit reverse electron-flow
decisions can be executed and audited. Its sequential-rollout failures show why
mechanism generation should not be a mandatory inference path for the present
transfer experiment.

The useful result from #81 is therefore diagnostic: sequential mechanism
rollout can accumulate errors. It is not the endpoint predictor used here.


## Continuity with the original Qwen3-8B MechET

The reliability study keeps the original MechET prediction contract in view.
The earlier Qwen3-8B MechET was not an independent precursor decoder with a
mechanism explanation appended afterward. It was the **inverse-reaction
policy** itself.

Starting from the product, the policy selected among three chemically distinct
decision families:

1. introduce a missing precursor-side fragment;
2. propose one coupled reverse electron-flow event;
3. terminate the trajectory.

The executor owned the molecular state. A successful electron-flow action
changed that state, and the precursor was derived only from the terminal
executed state. There was no independent precursor answer channel in the main
trace-owned method.

This distinction is retained in the reliability redesign. The central object
remains reverse electron-flow prediction; endpoint accuracy, hallucination
rate, calibration and route-level reliability are measurements of that
prediction contract rather than a replacement endpoint model.

## How missing environment / precursor-side molecules enter inverse inference

A product alone does not contain every atom required by its precursors. MechET
therefore has an explicit fragment-introduction operation.

### Original mapped-fragment interface

The original inverse builder partitions the final mapped reaction state into:

- the product components matching the product reference; and
- all remaining mapped components, recorded as root imports.

At training time those remaining components define the reference fragment
introductions. At inference time, however, the executor does **not** reveal or
retrieve those fragments. The policy must propose the fragment itself through
`import_fragment(fragment_smiles)`.

The executor then:

1. parses and canonicalizes the proposed mapped fragment;
2. requires every imported atom to have a unique positive map;
3. rejects atom-map collisions with the current state;
4. appends the fragment to the private molecular state;
5. marks the import as pending; and
6. commits the import to the trace only when a subsequent electron-flow event
   executes successfully.

A trajectory cannot terminate while imports remain pending. Thus the
environment validates and commits model-proposed chemistry; it does not
complete a missing precursor from the reference answer.

### Grounded unmapped-fragment interface

The later grounded-flow implementation removes atom-map prediction from the
model-facing fragment proposal. The policy proposes ordinary **unmapped**
fragment SMILES as part of the electron-flow event.

For a proposed event, the executor:

1. parses each unmapped fragment;
2. assigns fresh private atom maps internally;
3. enforces fragment-count and atom-count limits;
4. merges the fragments with the current private state;
5. grounds the proposed electron-flow roles against the complete event state;
6. executes the whole event transactionally; and
7. commits neither the imports nor the electron-flow move when the event is
   rejected.

The frozen implementation currently bounds one event to at most four imported
fragments, at most 64 imported atoms in total, and at most 24 heavy atoms per
fragment.

For replay supervision, reference fragments are scheduled to the **first
reverse electron-flow event that actually touches one of their atoms**. This
avoids introducing all precursor-side species at the beginning of a trajectory
when they are not yet needed.

### Reliability implication

Fragment proposal is a separate source of retrosynthetic hallucination from
electron-flow prediction.

A product-start failure can therefore arise because:

- the policy proposes the wrong missing fragment;
- the fragment is chemically/structurally invalid;
- the correct fragment is proposed at the wrong stage;
- the fragment is valid but the subsequent source/sink event is wrong; or
- the electron-flow sequence is individually legal but reaches the wrong
  endpoint.

These failure classes must be reported separately in the reliability study.
In particular, a wrong-fragment failure should not be attributed to the
electron-flow head.

## Position of the previous Qwen3-8B result

The previous Qwen3-8B MechET should be retained as the **sequential
trace-owned baseline**.

Its purpose was stronger than ordinary endpoint retrosynthesis: the policy had
to construct the precursor through model-proposed fragment introductions and
reverse electron-flow actions, while the executor controlled every state
transition and terminal endpoint.

This gives the old model two roles in the revised study:

1. **positive evidence for the contract:** on the earlier short-horizon
   replay-compatible subset, the executable formulation attained non-trivial
   precursor recovery with very high execution success;
2. **diagnostic evidence for the failure mode:** on the broader, longer
   trajectories, product-start performance degrades as fragment decisions and
   electron-flow decisions accumulate.

The new reliability work should therefore not erase or relabel the Qwen3-8B
model. It is the sequential MechET baseline against which the revised
factorization is judged.

The intended progression is:

    Qwen3-8B sequential MechET
        -> exposes fragment / source-sink / horizon failure modes
        -> revised reverse-electron-flow predictor
        -> lower hallucination at the reaction edge
        -> more reliable multi-step planning

The method remains MechET. A Jev-style typed decision implementation, if used,
is an architectural change for reducing decision and rollout error; it is not a
new scientific task or a replacement method name.

## Required fragment-specific reliability metrics

The reliability evaluation must add fragment-specific measurements rather than
folding every failure into endpoint exact match:

- fragment parse-valid rate;
- fragment first-use timing accuracy on trace-annotated data;
- reactive-fragment exact / recall;
- auxiliary-context fragment exact / recall;
- duplicate/repeated fragment proposal rate;
- unnecessary-import rate;
- pending-import termination failures;
- fraction of hallucinated reaction edges attributable first to fragment
  proposal versus electron-flow prediction.

For multi-step planning, every failed or pruned reaction edge should retain the
first assigned cause: fragment proposal, source/sink grounding, formal
execution, executable-wrong-successor, or termination. This makes it possible
to test whether improved route reliability comes from fewer fragment
hallucinations, better reverse electron flow, or both.
