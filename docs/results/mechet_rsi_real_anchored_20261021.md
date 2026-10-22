# PR92 real-data-anchored RSI correction

Status (2026-10-22): completed; GT precursor matching did not improve.
This is the explicitly requested PR92 bounded experiment, not a full-test result
or another SFT retraining. PR91 adapters, raw results and tasks are unchanged.

## Completed held-out results

The core endpoint objective was not improved by anchored DPO or the recursive
update: greedy GT structural matching is **6/32 for R0, 5/32 for M3 and 4/32
for R2v1**. ReplayOnly reaches 7/32, one additional target; this single-seed,
32-target comparison does not establish a reliable gain. Higher frozen-F0
round-trip confirmation must not be reported as higher precursor accuracy.
These results do not justify claiming successful endpoint optimization or
scaling this training configuration on the strength of round-trip gains alone.

All seven actors use the same fixed VALID32 products, original source-graph
initialization, prompts, actor-independent seed17 decoder and frozen F0.
The primary repeat-rejection guard is OFF. Greedy K1 has 32 episodes per actor;
stochastic K2 has 64. All 32 targets are present in every primary evaluation;
no TEST examples were used. Stochastic either-of-two is a coverage measure,
not an NLL-ranked top-1 result.

| Actor | Description | GT structural, greedy /32 | GT structural, either-of-two /32 | Productive round-trip, greedy /32 | Productive round-trip, stochastic first /32 | Productive round-trip, either-of-two /32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| R0 | Original Stage-II | 6 | 4 | 17 | 12 | 19 |
| R1v0 | Historical unanchored DPO, re-evaluated with v1 inputs | 2 | 5 | 6 | 7 | 9 |
| M1 | ReplayOnly | 7 | 3 | 18 | 13 | 17 |
| M2 | CorrectedDPOOnly | 5 | 2 | 17 | 9 | 14 |
| M3 | Anchored DPO + replay, round 1 | 5 | 2 | 17 | 10 | 16 |
| R2v1 | M3 + fresh D1 anchored update | 4 | 5 | 20 | 11 | 17 |
| R2staticD0 | M3 + original D0 anchored update | 4 | 3 | 19 | 11 | 17 |

GT structural counts use the evaluator's `reference_structural_targets`;
they are not an independent chemical-validity verdict. Productive round-trip
requires executed task-compatible progress and frozen-model forward
confirmation, not experimental evidence that a reaction occurs. R2v1's
greedy round-trip gain (17 to 20) does not rescue its lower GT matching (6 to
4); its stochastic coverage also falls from 19 to 17. Fresh versus static
round 2 differs by only one greedy confirmed target and ties at K2. Neither
comparison establishes recursive self-improvement.

| Actor | Terminal episodes, greedy /32 | Terminal episodes, stochastic /64 | Reverse tokens, greedy | Reverse tokens, stochastic |
| --- | ---: | ---: | ---: | ---: |
| R0 | 26 | 60 | 49,647 | 70,019 |
| R1v0 | 13 | 43 | 100,446 | 144,668 |
| M1 | 29 | 60 | 33,647 | 65,770 |
| M2 | 26 | 62 | 50,166 | 73,202 |
| M3 | 28 | 63 | 45,412 | 63,280 |
| R2v1 | 27 | 60 | 39,557 | 71,275 |
| R2staticD0 | 27 | 60 | 43,440 | 76,027 |

These token totals are reverse generation only, not end-to-end cost including
F0. The machine-readable snapshot also retains per-product token-matched
comparisons, all tool-error counts and the separate R0/M3 guard-on ablation;
guard-on outcomes are not substituted into the primary table.

### Actual updates and recursive budget

| New actor | Updates | Preference pool: pairs / sources | Scored assistant tokens | Active-loss tokens | Optimizer-stage seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 | 12 | 9 / 7 (DPO weight zero) | 6,898 | 1,322 | 20.24 |
| M2 | 12 | 9 / 7 | 6,898 | 5,576 | 23.83 |
| M3 | 12 | 9 / 7 | 6,898 | 6,898 | 26.42 |
| R2v1 | 12 | 19 / 13 | 6,689 | 6,689 | 36.44 |
| R2staticD0 | 12 | 9 / 7 | 6,898 | 6,898 | 23.40 |

The round-1 controls share the exact scored-example plan, not equal
active-loss tokens. Each arm uses only 12 actual replay examples, not the
entire 512-source replay pool. D0v1 uses 64 fresh F0 checks to obtain nine
strong pairs (seven IMPORT, two FLOW). Fresh D1v1 is collected from the saved
M3 adapter: 926 new episodes, 64 fresh F0 checks, 19 strong pairs (16 IMPORT,
three FLOW). R2v1 and static-D0 both start at M3 with fixed original R0 as the
probability reference. Both have 12 updates and a 6,898-token scored cap;
R2v1 spends 6,689 and leaves 209 unused. Identical token caps are not identical
realized token costs. F1 was not trained.

All five new adapters were saved; their bytes match the recorded SHA256s.
The completion file reports `finished`. The platform reports task `_02` as
`END`, `IsSuccess=true`, ending 2026-10-22 03:59:44 (UTC+8); creation was
2026-10-21 22:00:43. Roughly six hours is total task time including collection
and evaluation, not the short optimizer stages above.

The [final evidence snapshot](mechet_rsi_real_anchored_final_20261022.json)
contains the unabridged paired metrics, token-matched views, completion and
training manifests, recursive budget, guard-ablation summaries, and raw-file
paths/sizes/SHA256s. Full prompt/action/state journals and adapters remain at
the absolute artifact root below; large raw journals and model weights are
not embedded in Git. Historical S0 evidence is preserved unchanged.

## Measured S0 audit

| Historical pool | Reactions | Episodes | Original pairs | Corrected strong pairs | Sources | Tool signal |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v0 D0 | 128 | 990 | 11 | 4 | 4 | 3 IMPORT, 1 FLOW |
| v0 D1 | 128 | 966 | 20 | 17 | 15 | 17 IMPORT |

Reconstructed labels use the stored independent F0 results, mapped recorded net
classes and actual common-prompt action records. They do not turn one F0
mismatch into a hard chemical negative. D0 has 27 historical forward-hit
episodes, of which two fail task compatibility; D1 has 26, of which one fails.
These are episode counts, not independent reaction accuracies. The phosphate
and N-oxide fixtures reject only their terminal acid/base shortcuts relative
to the recorded structural tasks; genuine acid/base, redox and stereo remain
supported. The 17 IMPORT-only D1 pairs do not demonstrate FLOW-policy learning.

Full numerical source hashes, per-pair reasons and latest historical R2 metrics:
[S0 evidence](mechet_rsi_real_anchored_S0_20261021.json).
Original full records remain outside Git under
`outputs/agent/mechet_rsi_roundtrip_smoke_20261021/`.

## Real replay and initialization correction

512 independent TRAIN sources were checked against their actual Stage-II
records. All 3,666 original actions executed and all original prompts/history
and result states matched. The retained replay pool contains 1,995 complete
IMPORT/FLOW/FINISH calls, 1–4 per reaction, including assistant EOS and excluding
user/tool tokens from loss. No VALID or TEST row enters replay.

The exact mapped source product must come from the authenticated
`flower_inverse_tool_sft_action_delta_v1` parent, not a fresh mapping of a
canonical proof product. Two observed differences matter:

- Some source products retain explicit H graph nodes absent in the proof view.
- Even when the public prompt matches, replacing private atom ordering can
  select a different aromatic Kekule form. For train_110212 the same recorded
  arrows then produced a different, fragmented structure. The regression test
  reproduces failure from remapping and success from the original source graph.

In the frozen PR91 TRAIN128, 38 root prompts (and 8 of VALID32) differ from the actual Stage-II
source when initialized via the old normalization. For v1 **all actors** use
the same authenticated original product graph, with maps private and only the
existing map-free product/state/inventory/history visible. No expected
precursor, future action, task class or horizon enters a policy prompt.
The complete TRAIN128 + VALID32 root-prefix gate passed 160/160 locally.

Because this fixes input representation as well as decoding seeds, original
v0 numbers are historical diagnostics, not a clean isolated weight ablation.
R0 and v0 R1 are re-evaluated alongside the new controls under the same v1 input
and decoder contract. D0v1 is generated afresh; original D0 is used only in S0.

## Executed implementation

- Within-target common-state complementary forward allocation, 64 fresh checks
  per TRAIN128 cohort; cached checks do not consume new budget. A terminal
  versus truly rejected same-prompt action may receive one needed check.
- Strong preferences require terminal executed task progress + independent F0
  hit. Pairs have exact prompt IDs/full action EOS, distinct chemical
  successors and at most two contributions per source. Unknown stays unknown.
- M1 ReplayOnly / M2 corrected-DPOOnly / M3 anchored DPO+replay all start from
  original R0. Fixed R0 probability reference, beta 0.1, LR 2e-6, replay weight
  0.25; 3 preference + 1 authentic replay slots, at most four uses per unique
  pair and 20 updates derived from actual pool size. Zero pairs do not cause
  fabricated DPO updates; replay-only remains available.
- Controls share the same scored-example/update plan. Active-loss tokens
  differ by loss ablation and are reported separately. Each branch's actual
  gradient norm, weight change and saved adapter SHA are logged.
- Greedy K1 and stochastic K2 are separate. Decode seeds depend on target,
  candidate, decision and seed17, not actor SHA/phase. Actor SHA stays in cache
  identities. Per-product token censors retain the complete VALID32 denominator.
- Repeat-rejection guard is off for the primary comparison and on for BOTH R0
  and M3 in a separate runtime ablation. No rejected call is marked accepted.
- Fresh D1v1 uses updated M3 only, conditional on a TRAIN-only replay retention
  gate. Recursive and static-D0 arms share an optimizer count and scored-token
  cap; actual spent and unused tokens are disclosed because complete calls are
  indivisible. This is not identical realized token consumption. F1 is deferred.

One fresh-context branch review found and corrected completed-checkpoint reload,
cached-allocation starvation and terminal-versus-invalid coverage bugs. Their
regressions were observed failing before fixes. CPU unit tests use actual tiny
Qwen/LoRA updates, not mocked loss gradients; they are implementation evidence,
not experimental chemical results.

## Reproduce / artifacts

```bash
python -u scripts/run_mechet_rsi_anchored.py \
  --config configs/experiments/mechet_rsi_real_anchored_resolved_20261021.yaml
```

Adding `--prepare-only` executes the entire model/data/replay/root-input
preflight without creating a GPU engine. That actual local command passed:
512 TRAIN replay reactions, 1,995 retained actions, all 160 root prefixes.

### First launch failure (preserved, not an optimizer result)

Task `meteor_mechet_pr92_anchored_rsi_2a100_qy_20261021_01`, instance
`8b1d899ea13fdea001a15ebd00a74061`, pinned code `0a1100d`, acquired two
A100-SXM4-40GB cards in QY. Ceph mounted and Taiji default stdout showed the
45-second heartbeat, Torch2.6.0+cu118, vLLM0.8.5 and RDKit2026.03.4.
It exited before engine creation or optimizer work: the real proof validation
split is named `val`, which the new source gate incorrectly rejected as if it
were TEST. The gate now normalizes only `val` to `valid`; TEST remains refused.
A regression reproduced the failure and the full no-GPU launch preflight then
passed. The failed instance is END; the separate pinned retry below completed.

### Corrected submission (now completed)

Retry task: `meteor_mechet_pr92_anchored_rsi_2a100_qy_20261021_02`;
instance `8b1d89f5a13f013001a15ec3a539449c`, two ordinary A100 cards in
`AI4LSLab_KEMO_QY_A100`, location `qy`. Runtime is pinned to
`f77d60cd28a882c6f0e840ea302c49fbd7149d4f`; public submission config is
`configs/taiji/meteor_mechet_pr92_anchored_rsi_2a100_qy_20261021_02.json`.
Final pre-submit suite: **540 passed, 1 skipped**, 9 PEFT warnings.
Default terminal logs retain all progress plus a 45-second heartbeat. The
private Ceph init is recovered from a prior successful task, not stored in Git.
Creation/start success alone was only submission evidence. Completion and the
measured validation results are now recorded above.

Observed live after retry: both A100-SXM4-40GB cards are present, Ceph is mounted,
and the real vLLM owner plus HF scorer loaded the pinned models. The source
preflight passed in the GPU container and `provenance.json` records `f77d60c…`.
At the historical first sampling readback, D0v1 K2 had **871 durably saved action records,
99,694 generated tokens, 800 accepted actions, and 7/256 completed episodes**
(all seven terminal). This is a partial TRAIN collection, not endpoint accuracy
or evidence of improved weights; controls and held-out evaluation were pending
at that time, and have since completed.
Worker stdout/stderr were verified to have the same pipe targets as PID1.
The Taiji log API was serving a lagging bootstrap snapshot while raw journals
and phase events continued growing; logs are not redirected to a separate file.

New isolated root (absolute):
`/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/mechet_rsi_real_anchored_smoke_20261021`

Exact raw prompts/token IDs/action arguments/states/errors are durably retained
in `*.jsonl.actions.jsonl`; complete episodes in `*.jsonl`, forward bridges and
labels in `*.feedback.json` / `*.bridge_audit.json`. Gates and provenance are
`prompt_parity.json`, `provenance.json`, `replay/manifest.json`, and `S0/audit.json`.
Actual updates appear under `models/*/manifest.json`, per-step losses/gradients
under `models/*/training_steps.jsonl`, and held-out outcomes incrementally under
`evaluation/paired_validation.json`. A finished run requires `completion.json`;
submission or a falling training loss is not completion or evidence of improvement.

No PR is merged and no paper/Graph/CRN file is modified. R0 remains
`40204c7a…7e9de75`; F0 remains provisional checkpoint-54000
`38e663a3…aa89e`, not a silently substituted final forward checkpoint.
