# JevRetro: typed single-step retrosynthesis

## Scope

This is a clean spin-off from PR #81. PR #81 remains the mechanistic evidence
branch for typed chemical decisions. JevRetro tests the standard retrosynthesis
question directly:

> Given one product molecule, can a non-autoregressive typed decision model
> produce a ranked Top-K set of precursor structures?

The primary metrics are conventional Top-1 / Top-5 / Top-10 structural
precursor exact-match accuracies. Electron-flow next-state accuracy is not
substituted for this endpoint metric.

## Benchmark order

### Tier 1: USPTO-50K

Use the frozen Schneider split already produced by
scripts/prepare_uspto50k_benchmark.py:

- train: 40,008
- validation: 5,001
- test: 5,007

This is the first standard single-step benchmark because it has a stable
product/precursor contract and existing MechET evaluation infrastructure.

### Tier 2: FlowER-retro endpoint view

Use the complete reaction-level endpoint view:

- train: 257,171
- validation: 2,890
- test: 28,971

No executable-trace filtering is allowed. Mechanistic traces may be used only
as auxiliary analysis or an auxiliary loss after the endpoint model itself is
defined on the complete endpoint train split.

mech-USPTO-31k is not the first JevRetro endpoint benchmark because the current
project has already exposed product-field provenance ambiguity there. It remains
useful for mechanism and transfer analyses after the standard endpoint model is
established.

## Phase 0: full-denominator typed-target audit

Run the compiler on every row of train/valid/test:

    python scripts/build_jevretro_endpoint_decisions.py \
      --train  <train endpoint jsonl> \
      --valid  <valid endpoint jsonl> \
      --test   <test endpoint jsonl> \
      --output-dir outputs/jevretro/<dataset>/typed_v1

The compiler factorizes each mapped product-to-precursor pair into:

1. atom-state edits on product-origin atoms;
2. bond-state edits between product-origin atoms;
3. explicit product-atom deletions if required;
4. precursor-only residual fragments represented as attachment templates with
   numbered dummy slots and product-atom anchors.

Every compiled program is immediately applied back to the product. The rebuilt
precursor must equal the frozen structural precursor under canonical map-free
isomeric SMILES. Rows are never silently dropped. If any row fails, the
manifest records training_allowed=false and the full failure ledger remains.

The audit also reports:

- bond-edit count distribution;
- atom-edit count distribution;
- attachment count and attachment-slot distribution;
- train attachment-template vocabulary size;
- valid/test attachment-template OOV rows, occurrences and unique templates;
- maximum decision cardinalities.

These quantities decide whether attachment prediction can be a closed
train-vocabulary decision or needs an open-vocabulary retrieval/generation
module. Do not choose that architecture before the full audit.

## Phase 1: Jev-style endpoint policy

The endpoint model must use the full endpoint training split, not only reactions
with executable mechanism traces.

Preferred architecture:

    mapped product
       |
       v
    shared small-LM encoder (Qwen3-0.6B + LoRA)
       |
       +--> typed bond-edit/set head
       +--> typed atom-state-edit head
       +--> typed attachment-count head
       +--> typed attachment anchor/template head
       |
       v
    deterministic endpoint reconstructor
       |
       v
    ranked precursor candidates

The representation should remain set-based or parallel wherever possible.
Do not turn the edit program back into an autoregressive text sequence.

The critical matched control is:

    Qwen3-0.6B autoregressive product -> precursor
    vs
    Qwen3-0.6B JevRetro typed decisions

with the same endpoint training IDs, base revision, data budget and evaluation
denominator. Existing 8B MechET and external retrosynthesis methods are field
references, not the only causal control.

## Phase 2: deterministic Top-K decoding

JevRetro must output a scored candidate set, not one autonomous trajectory.

Candidate score should be a frozen sum of typed log-probabilities, for example:

    score(program)
      = log p(edit_count)
      + sum log p(bond_edit)
      + sum log p(atom_edit)
      + log p(attachment_count)
      + sum log p(anchor/template)

The decoder applies each candidate program, sanitizes with RDKit, removes
invalid structures, canonicalizes and deduplicates. Invalid candidates remain
spent candidate budget; they are not replaced with gold-aware alternatives.

Evaluate with:

    python scripts/evaluate_jevretro_topk.py \
      --references <test endpoint jsonl> \
      --predictions <ranked candidate jsonl> \
      --output outputs/jevretro/<dataset>/topk_report.json

Prediction rows contain a stable id and candidates with precursor and score.
The evaluator keeps the full reference denominator and reports ranked
Top-1/5/10.

## Promotion gates

A single-step JevRetro claim requires all of the following:

1. full train/valid/test endpoint decision compilation is audited;
2. no test label influences option vocabulary selection or ranking;
3. Top-1/5/10 are measured on the standard full denominator;
4. the 0.6B autoregressive matched control is available;
5. inference latency, peak memory and candidate count are reported;
6. test is evaluated only after architecture/model selection is frozen on
   validation.

The old PR #81 16.7% product-proxy rollout is not a JevRetro single-step
baseline and should not be compared to these Top-K results.

## Multistep follow-up

Multistep synthesis planning begins only after the single-step model is frozen.
The same ranked candidate provider will implement the existing Syntheseus
BackwardReactionModel interface. Retro* search then compares models under
matched reaction-model-call and wall-time budgets.

The intended multistep metrics are:

- solved-target rate;
- reference-route Top-1/5/10 where defined by the benchmark;
- reaction-model calls to first solution;
- wall time to first solution;
- route length;
- route diversity;
- fraction of proposed edges rejected by deterministic chemistry checks.

This second stage should use a recognized route-planning benchmark and its
matched one-step training distribution rather than reusing mech-USPTO-31k by
convenience.
