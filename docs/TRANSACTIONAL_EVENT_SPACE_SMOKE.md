# GT-independent transactional event-space smoke

Date: 2026-09-12

This pilot separates state-derived formal support from learned chemical ranking.
The executor enumerates electron sources from the current private mapped state,
then generates only topology-compatible sinks conditional on a selected source.
The product answer, proof and reference event are not inputs to enumeration.
Reference moves are consulted only afterward to measure coverage.

## Frozen-data coverage smoke

Input: the first 1,000 events in the frozen FlowER validation trace view.

| Measure | Result |
|---|---:|
| Event coverage | 997 / 1,000 (99.7%) |
| Ordinary/radical move coverage | 2,008 / 2,008 (100%) |
| Uncovered events | 3 `BE_DELTA` events |
| Polar source candidates, median / P95 | 58 / 148 |
| Gold-source-conditioned sink candidates, median / P95 | 71 / 183 |

Radical-pair candidates are maintained as a separate family because mixing all
metal/halogen pairs into ordinary polar source selection makes the displayed
inventory unnecessarily large. A production policy should first select the
event family and then rank candidates inside that family.

Artifact:
`outputs/eval/transactional_event_space_valid1000_20260912/evaluation.json`.

## Unadapted Qwen forward-versus-inverse judgment

The same 32 short validation events with exactly four formally executable
gold-derived diagnostic options were scored by the unadapted Qwen3-8B model.
This is intentionally only a capability diagnostic: the option construction is
not deployable because it includes a reference event and gold-derived hard
negatives.

| Direction shown to Qwen | Top-1 | Mean gold rank | Random Top-1 |
|---|---:|---:|---:|
| Inverse next event | 7 / 32 (21.9%) | 2.438 | 25.0% |
| Reversed forward event | 9 / 32 (28.1%) | 2.375 | 25.0% |

Only two additional cases become correct under forward presentation. This does
not support replacing training with zero-shot forward judgment. Forward
orientation remains useful as a scoring representation, but it needs supervised
contrastive adaptation on executable hard negatives.

Artifacts:
`outputs/eval/forward_judge_smoke_20260912/{inverse,forward}/evaluation.json`.

## Claim boundary

The coverage smoke establishes candidate-support feasibility, not complete-event
ranking or endpoint accuracy. The Qwen smoke establishes that an unadapted model
is near random even when options are already formally executable. A deployable
next stage must construct candidates without references, train their chemical
ranking, separately generate first-use imported fragments, and retain the full
28,971-row FlowER endpoint denominator.
