# NMI compositional-generalization AutoResearch

## Purpose

This campaign tests one paper-level hypothesis without adding another model family:

> **Reusable local electron-flow operators can be recomposed into unseen complete reaction programs when each decision is grounded in the molecular state actually produced by execution.**

The campaign is deliberately narrower than the vNext roadmap. It does not introduce Tree-RL, a learned world model, a new graph proposer, a larger backbone, or a new synthesis planner.

## Prior-work audit

The implementation line preceding this campaign is already merged:

- PR #70 — protocol-v2 train/inference parity repair.
- PR #71 — vNext roadmap.
- PR #72 — coupled source/sink pointers, structured actions and tree-credit infrastructure.
- PR #73 — frozen AutoResearch controller.
- PR #74–#78 — runtime isolation / CPU-thread / XGrammar / vLLM schema compatibility repairs.

The scientific status is not identical to merge status:

- coupled pointer **oracle-state localization** is established (pair R@1 56.5% -> 91.8%; pair R@4 95.4% -> 98.9%);
- packing has already ended as **SCIENTIFIC_STOP** and is not retried here;
- product-start pointer 16 -> 256 -> 1,319 and the final five-mode runtime summary are not promoted as scientific results until their frozen reports exist;
- no unresolved vNext engineering result is allowed to redefine the compositional experiment.

The NMI compositional study therefore uses the frozen paper model contract and existing deterministic executor. vNext pointer/runtime improvements are supporting engineering only.

## Scientific questions

### Q1 — Is there a reusable local operator basis?

Measure how often held-out reactions contain:
- an unseen complete execution composition;
- only primitives seen in training;
- at least one unseen primitive.

The key observable is the size of the **program-unseen / all-primitives-seen** region.

### Q2 — Does execution matter for recombination?

Train/evaluate matched Qwen3-8B conditions on the same frozen reaction IDs:

1. **endpoint-only** — product -> precursor;
2. **open-flow** — one-shot/open-loop electron-flow program;
3. **closed-loop MechET** — action -> execute -> authoritative state -> next action.

The backbone revision, LoRA rank, seed, reaction IDs and candidate budget are frozen. Differences in maximum sequence length are reported rather than hidden.

### Q3 — What training familiarity predicts success?

For every evaluation reaction compute:
- complete-program frequency in train;
- minimum primitive frequency;
- mean log primitive frequency;
- fraction of primitives seen;
- scaffold / reaction-centre overlap when available.

The main analysis asks whether closed-loop success remains associated with primitive familiarity when complete-program overlap is zero.

### Q4 — Does the same relation transfer externally?

PMechDB/PMechRP are optional external validation stages. They are manual/request-gated upstream assets and the campaign never bypasses that gate.

## Experiment DAG

```text
old-P0 status audit
        |
operator-basis audit
        |
strict MechComp split + structural-overlap audit
        |
matched control materialization
        |
  +-----+------------------+
  |     |                  |
endpoint-only        open-flow        closed-loop
 train/eval          train/eval        train/eval
  +-----+------------------+
        |
familiarity x program-novelty analysis
        |
external PMechDB/PMechRP (only after explicit data unlock)
```

## Frozen split

Primary basis:
`source_to_sink_execution_moves_v1`.

The strict test condition must satisfy:
- non-empty test set;
- zero train/test complete-composition overlap;
- every test primitive observed in train;
- structural-overlap audit complete;
- near-duplicate rate within the declared tolerance.

The existing `build_mechcomp_ood.py` and `src/mechet/proof_splits.py` remain authoritative for split construction.

## Matched conditions

All three controls are materialized from the same stable IDs selected by the strict split.

The campaign must fail closed if any source representation cannot provide exactly the same stable-ID universe.

The primary comparison is **open-flow vs closed-loop** on program-unseen / primitive-seen reactions. This isolates executed state feedback from explicit electron-flow notation.

## Primary metrics

For each condition:
- StructuralEndpointPass@1/5/10;
- formal execution where defined;
- successor-chain success where defined;
- missing predictions retained as failures.

For the familiarity analysis:
- success by primitive-frequency quantile;
- success by complete-program frequency;
- success on program-unseen/all-primitives-seen;
- success on any-primitive-unseen negative-control rows when available;
- scaffold/reaction-centre strata;
- reaction-level bootstrap intervals.

## Promotion gates

### Data gate
At least 100 program-unseen/all-primitives-seen test reactions are required for a headline compositional analysis. If fewer exist, stop and report the data limitation.

### Closed-loop gate
On the strict program-unseen/all-primitives-seen set, closed-loop MechET must outperform open-flow in the preregistered primary endpoint or successor-chain metric with a positive paired direction. Confidence intervals are reported; no threshold is retuned after observation.

### Mechanistic generalization gate
The closed-loop advantage must not be explainable solely by exact/scaffold/reaction-centre overlap. The familiarity analysis must therefore be reported jointly with the structural-overlap audit.

### External gate
External datasets are validation only. Failure does not trigger architecture changes inside this campaign.

## Explicit stop rules

Stop/narrow the claim if:
- strict split is empty or too small;
- a held-out primitive is accidentally required for the headline C2/C3 claim;
- matched conditions use different reaction IDs;
- a control checkpoint was trained on the pre-split/full dataset;
- test data are accessed before checkpoint/metric freeze;
- structural near-duplicates dominate the held-out set;
- open-flow and closed-loop differ in backbone revision or candidate budget.

## AutoResearch contract

Use the existing finite-state controller:

```bash
export MECHET_TAIJI_CLIENT=/path/to/taiji_client
export MECHET_TAIJI_DONOR_TASK=<successful-donor-task>
export MECHET_AUTORESEARCH_CODE_MIRROR=/aaa/fionafyang/buddy1/whaleywang/MechET-autoresearch-nmi-compositional-20261002

python scripts/run_vnext_autoresearch.py \
  --campaign configs/autoresearch/nmi_compositional_20261002.yaml \
  --ledger outputs/autoresearch/nmi_compositional_20261002/ledger.json \
  --workdir outputs/autoresearch/nmi_compositional_20261002 \
  --mode run
```

The controller may submit/poll/retry declared infrastructure failures and apply frozen gates. It may not alter hypotheses, thresholds, data, seeds, architecture or test access.
