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
