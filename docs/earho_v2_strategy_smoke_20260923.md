# EARHO v2 strategy smoke, 2026-09-23

Scope: mech-USPTO-31k **current-compiler executable trace view** (10,152 train / 1,319 valid / 1,253 test), not the complete 24,959 / 3,120 / 3,120 reaction denominator. The five-round EARHO campaign sampled 128 distinct train products per round and used a fixed 128-reaction validation monitor. This is strategy screening, not paper test-set evidence. No test rows were loaded.

Paper-aligned training contract: product-start rollout; first consequential divergence from independently replayed reference; same-anchor K=8 executed successors; invalid/no-op/cycle rejection and successor-equivalence pooling; positive only for reference-equivalent successor or exact endpoint; all-negative groups receive zero policy advantage and verified replay; learned successor P/N critic; adaptive continuation horizon. The monitor used product-only, full-episode K=2 sampling and beam width 2.

## Completed training rounds on the fixed monitor

| Policy | Strict endpoint candidates | Formal execution candidates | Pass@2 reactions |
| --- | ---: | ---: | ---: |
| Stage-II parent (before EARHO) | 0/256 | 0/256 | 0/128 |
| EARHO round 0 | 7/256 | 69/256 | 6/128 |
| EARHO round 1 | **21/256** | 120/256 | **21/128** |
| EARHO round 2 | 17/256 | 131/256 | 16/128 |
| EARHO round 3 | 18/256 | 141/256 | 17/128 |
| EARHO round 4 | 18/256 | 98/256 | 15/128 |

The predeclared selection rule in the campaign runner is candidate exact-endpoint rate on the fixed monitor. It selected round 1. Later rounds raised formal execution but did not improve exact endpoints. The monitor has been reused for selection, so 21/128 is not an unbiased generalization estimate.

## Matched strategy probes using the frozen round-1 actor

| Search condition | Strict endpoint candidates | Formal execution candidates | Pass@2 reactions | Decision |
| --- | ---: | ---: | ---: | --- |
| Learned critic, beam 2 (control) | **21/256** | 120/256 | **21/128** | Retain provisionally |
| No critic, beam 2 | 22/256 | 135/256 | 20/128 | No reliable endpoint gain; do not adopt |
| Learned critic, beam 4 | 15/256 | 120/256 | 12/128 | Reject |

All three probes use the same actor weights, 128 reaction IDs, K=2, seed, full-episode executor, decoding settings and endpoint scorer. Critic removal gained nine reaction-level hits and lost ten relative to control. Beam 4 gained seven and lost sixteen relative to beam 2. These numbers support retaining beam 2, not a broad claim that a learned critic is beneficial.

The no-critic condition produced 113 formally terminated but wrong candidates among 256, so merely increasing the formal execution rate is not the desired optimization target. In particular, forcing extra electron steps or choosing a terminal with reference information at inference would alter the product-only protocol and was not done.

## Confirmation running

`meteor_mechet_earho_v2_round01_full_valid_8a100_qy_20260923_01` evaluates the selected round-1 actor and critic with beam 2 and K=2 on all 1,319 executable-view validation reactions. Its pinned input manifest is `outputs/eval/earho_v2_round01_full_executable_valid_20260923/plan.json` under the main repository. Report all 1,319, plus the 1,191 reactions not in the 128-reaction selection monitor. The candidate run is not yet a full 3,120-reaction valid evaluation or a 3,120-reaction test evaluation.

Do not call K=2 stochastic Pass@2 Top-2. Do not promote any setting to a main result until the larger validation confirms endpoint improvement; keep formal execution, wrong terminal, and failure categories as secondary diagnostics.
