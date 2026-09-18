# Active ICLR status

## 2026-09-18: natural-language protocol v2 repair

Keep the agreed mechanism-first method.  The current implementation authority
is `docs/NATURAL_LANGUAGE_PROTOCOL_V2.md`: one identical inventory-bearing
observation for all three tool choices, inference-time private-map mirroring in
the builder, executor-owned successor states, legal repeated imports, and
40-decision/32-import coverage.  Full validation gold replay is 2,890/2,890 and
prompt parity is 22,341/22,341.  Existing v1 history training/smoke remains
historical diagnostic evidence and must not be reported as v2.

The incorrect running v1 history task has now been stopped.  Full remote v2
conversion first found 15 train / 0 valid / 7 test Kekule-drift rows and
correctly prevented training.  Commit `408e132` repairs the fallback delta and
all 22 rows now replay exactly.  Replacement task
`meteor_mechet_nl_protocol_v2_full_sft_2stage_8h20_zjk_20260918_02` (instance
`8b1d8014a0af10a701a0b3501c720673`) is running on 8xH20.  It performs the full
zero-unresolved data gate, then clean State-SFT from the pinned Qwen3-8B base,
then compact-history Trajectory-SFT; submission/running state is not evidence
that either training stage has started or completed.

> This file is the short-lived execution authority for the current ICLR work.
> `PROJECT_MEMORY.md` remains authoritative for historical dataset/model lineage.
> If an older operational note in `PROJECT_MEMORY.md` conflicts with this file,
> this file wins for **current experiment priority and A7 observation choice**.

Last updated: 2026-09-17.

## Current three-stage method decision

The active algorithm is now described as:

1. **State-SFT** for executable local inverse transitions;
2. **compressed-history Trajectory-SFT** for causal trajectory-prefix
   conditioning without repeated full-state transcripts;
3. **Execution-Anchored Receding-Horizon Optimization (EARHO)** for adaptive
   frontier-local policy improvement under real executor transitions.

EARHO replaces the informal label “hard-case RL.”  It identifies the first
recoverable divergence frontier, pools alternative actions by executed chemical
successor, assigns no positive advantage to all-negative groups, uses verified
transition replay when policy support is absent, and expands the optimization
horizon only as competence advances.  Reference successors and endpoints are
training-private and never enter product-only inference prompts.

The full algorithm contract and evidence gates are in
`docs/EXECUTION_ANCHORED_RECEDING_HORIZON_OPTIMIZATION.md`.  Historical
`anchor_branch` and `successor_horizon` identifiers remain unchanged for run
lineage.

## Natural-language anchor-branch post-training (v4 correction gate)

The bounded productive task ended during its pre-update validation after
228/512 planned candidates (114/256 reactions), so it produced no RL adapter
and must not be reported as a completed experiment. The preserved partial
baseline had 172/228 formal terminals, 45/228 productive terminals, 127/228
unchanged-target terminals, and 1/228 exact endpoints. The exact hit was a
genuine full product-only trajectory; the dominant failure was instead the
action/import branch appending context and finishing without transforming the
target.

PR #59 now carries the v4 implementation correction. The scientific method is
unchanged: exact executor reset, same-state branching, executor verification,
local first-action credit, value-ranked gold-free continuation, and curriculum
toward full product-only episodes. The correction rejects unchanged-target
finish, preserves explicit-hydrogen import participants, adds private
reference-successor credit,
materializes one verified replay record per train reaction, and preserves the
evaluation denominator across isolated collector exceptions. The next gate is
`configs/agent/natural_language_anchor_branch_rl_verified_replay_smoke_a100.yaml`
(64 train reactions, K=8, 32 validation reactions); do not restart the larger
2,560-reaction campaign until this gate completes. It was submitted to ordinary
Qingyuan 8xA100 as
`meteor_mechet_nl_anchor_branch_rl_verified_replay_smoke_8a100_qy_20260917_01`
(instance `8b1d813ea0a4c8ab01a0ad78ebbc0db3`). At the first post-submit check it was
still `PENDING`; submission success is not counted as execution success.

The endpoint-shaped v2 smoke completed successfully.  Its train rollouts had
315/512 formal terminals and one exact endpoint, but that exact candidate was
an H=1 suffix after 7/8 reference events.  Fixed full-episode validation stayed
24/32 formally terminal and 0/32 exact before and after the update.  Of the 24
post-update wrong terminals, 16 retained the unchanged target and appended
fragments.  This is an executable no-op, not a retrosynthetic solution.

Commit `f6bff25` adds a no-transform penalty, a separate productive-execution
metric and persisted per-decision continuation traces.  The full bounded task
`meteor_mechet_nl_anchor_branch_rl_productive_full_8a100_qy_20260916_01`
(instance `8b1d813ea0a4c8ab01a0aaabe7a30977`) uses 10 x 256 distinct train
reactions, K=8, 50% full episodes and a disjoint 256-reaction validation
monitor.  This is not a 257,167-row epoch.  At the initial live check the
ordinary Qingyuan 8xA100 Pod and Ceph mount were present, while the pinned vLLM
runtime was still staging and GPUs were idle; check collectors and GPU memory
before claiming rollout progress.

The current post-training actor is the completed natural-language electron-
event SFT, not the historical Python-template checkpoint. PR #59 implements
executor reset states, K=8 same-state first-tool branches, successor-state
equivalence pooling, endpoint-primary reward and first-tool-only local credit.
The bounded 8xA100 task was
`meteor_mechet_nl_anchor_branch_rl_smoke_8a100_qy_20260916`, instance
`8b1d813ea0a4c8ab01a0a9893c1f075d`. It ended successfully after 16 optimizer
updates. Validation execution improved from 16/32 to 20/32 candidates, but exact
endpoint success remained 0/32; the original SFT therefore remains the selected
adapter. Test was unused.

The failure is traced to sparse reward: only 1/512 training candidates was an
exact endpoint, while 256 wrong terminals all received zero and 255 invalid or
incomplete candidates received -0.1. The prepared v2 repair retains exactness as
the only positive outcome and adds map-invariant, heavy-atom-weighted endpoint
similarity plus first-successor progress to rank non-exact branches. All wrong
terminal and invalid outcomes remain negative. Twenty-five relevant tests pass, and
offline rescoring yields 218 reward values with contrast in 96/128 prompt-mode
groups. Continuation ranking now uses the frozen state-value-v2 adapter over
executor-valid unique successors instead of comparing raw NLL across the action
and event prompt formats. The v2 8xA100 smoke was submitted as
`meteor_mechet_nl_anchor_branch_rl_shaped_smoke_8a100_qy_20260916`, instance
`8b1d81eea0a4bdf601a0aa2e28e108b0`. Live inspection confirmed
`TRAINING_RUNNING`, the private Ceph mount, the wrapper/heartbeat processes and
local vLLM-runtime staging. GPU utilization is expected to remain zero until
that one-time copy finishes; do not report optimizer activity before collector
processes and GPU allocation are observed. See
`docs/NATURAL_LANGUAGE_ANCHOR_BRANCH_RL.md`.

## Natural-language electron-event candidate (active)

- Branch/implementation: `feature/natural-language-electron-sft-20260913`
  (`34074c7`).
- Source denominator: the complete strict-executable FlowER universe,
  `257,167 / 2,890 / 28,967`; this is not the 32k proof subset and must not be
  called unqualified “FlowER full”.
- Observation: target plus the executor's current unmapped SMILES. The
  environment annotates each current atom with a temporary per-step `Axx`
  alias; private atom maps and the next state are not model-visible.
- Prediction: one import, one natural-language retrosynthetic electron-flow
  event, or finish decision. The executor resolves aliases, applies the event,
  and returns the authoritative next state.
- Reproducibility: conversion and replay are pinned to RDKit `2026.03.4`.
  The 256-reaction smoke and complete validation split replayed with zero
  unresolved reactions. Taiji additionally completed the full test split at
  `28,967 / 28,967` with zero unresolved reactions before starting train data
  conversion.
- Active Taiji task:
  `meteor_mechet_natural_language_event_sft_1ep_8a100_qy_20260913_03`, instance
  `8b1d8228a08afcb601a096e8cdb913e6`, normal non-elastic `8 x A100` in Qingyuan.
  Full conversion and distributed tokenization are complete; eight training
  ranks are running the one-epoch Qwen3-8B QLoRA SFT. The `_02` instance ended
  before optimization because duplicating both the 20.4-GB token cache and the
  model cache exhausted node-local temporary storage; `_03` keeps only tokens
  local and loads the pinned model from the shared read-only cache.

This is a candidate representation experiment. It does not supersede the
paper's current A7 choice until its frozen validation evaluation is available.

### Frozen checkpoint-14000 local diagnostic

- Frozen adapter SHA256:
  `3adfa321f1cc06c257dd96cc5dde1bd2e7b0c9608dc7cd11b54b3749e5b79f10`.
- Fixed seed-17 stratified validation sample: 256 reactions, 1,979 decisions
  (704 import, 1,019 electron event, 256 finish).
- Completed normal non-elastic task:
  `meteor_mechet_nl_event_valid256_ckpt14000_k1_8v100_cq_20260913_01`, instance
  `8b1d81f5a08afd1e01a09aa1993619a8`, `8 x V100` in Chongqing.
- This is greedy K=1 at authoritative current states. It reports tool choice,
  import matching, electron source/destination, strict execution, and mapped
  plus map-invariant chemical successor accuracy. It is explicitly not a
  product-only closed-loop or test-set endpoint result.
- All 1,979 planned decisions completed with zero missing or extra rows. Tool
  selection is 99.34%, argument compilation 98.23%, electron-event formal
  execution 96.57%, strict event match 68.01%, and map-invariant chemical
  successor match 73.31%. Reactive-fragment exact match is 41.21%, versus
  8.02% for endpoint-only context. Event exact match is 86.8%, 71.2%, 31.0%
  and 0/23 for one through four coupled flows; six four-flow predictions reach
  the same chemical successor via a shorter representation. These are fixed
  local diagnostics, not autonomous endpoint results.
- Scientific framing and evidence boundaries are consolidated in
  `docs/EXECUTABLE_INVERSE_DYNAMICS.md`.

## What is already on `main`

- PR #35 (`b7e26027`) merged the full-data protocol/runtime update.
- PR #40 (`4b410e52`) merged the code/protocol repair for Issue #36:
  matched mech-USPTO training/inference prompt + tool schema + 12-call budget,
  one generation worker per 40-GiB A100, explicit failed/OOM candidate accounting,
  NLL/first-candidate metrics, and frozen lineage hashes.

The old protocol-mismatched ~1.60% mech-USPTO action-delta result remains
**diagnostic only**.

## Active ICLR issue

### #36 — matched mech-USPTO action-delta rerun (closed)

Implementation/protocol repair is complete on `main` via PR #40.
Issue #36 was closed on 2026-08-26. Do not create, retry or expand experiments
under #36. The one already-running K=10 + NLL task may finish and be archived;
any newly discovered defect must be scoped as a separate issue.

### #37 — compact-full-state A7 main run

This is the only active A7 implementation/training issue.

Current decision:

- `compact_full_state`: **main A7 candidate**;
- `action_delta`: ablation / fallback;
- legacy `full_state`: historical reference;
- reaction-site-marked SMILES: outside the current ICLR run.

The paper method is unchanged: after each accepted action, the next decision
must receive the actual executor-produced current chemical state. The compact
format removes duplicate serialization, not state information.

Fast execution gate:

1. implement one `compact_full_state` observation mode;
2. on a 1k–5k frozen slice verify identical stable IDs, byte-identical assistant
   actions, chemically identical replayed states/endpoints, no dropped rows and
   no truncation;
3. require a substantial input-token reduction (target <=50–60% of legacy
   full-state);
4. immediately submit Qwen3-8B A7 for **1 epoch**;
5. run the cheapest valid frozen validation evaluation after epoch 1;
6. if performance clearly recovers from action-delta and approaches the
   full-state reference, freeze compact-full-state as A7 main and launch the
   full K=10 + NLL-ranked test; continue epoch 2 only if validation still shows
   clear headroom.

Do not wait for three new matched full trainings before obtaining the A7 result.

Gate status on 2026-08-26:

- 2,048 frozen training rows rebuilt with zero quarantine;
- stable IDs, assistant actions, every authoritative state and the executed
  endpoint are identical to the matching legacy full-state rows;
- no model-visible proof/digest or duplicated before/after state fields;
- Qwen3-8B chat-template input tokens are 11,951,517 vs 23,721,228, or
  **50.38%** of legacy full-state;
- full strict-universe train/valid/test construction is in progress; submit
  exactly one resumable 1-epoch A7 job after the full artifact audit passes.

## Consolidated / deferred issues

- #38 is superseded by #37. Its useful content (lossless removal of redundant
  state/proof serialization) is now part of the compact-full-state A7 plan.
- #39 reaction-site-marked SMILES is deferred beyond the current ICLR paper and
  must not block A7.

## Priority order

1. implement + launch #37 compact-full-state A7;
2. allow the already-running #36 matched rerun artifact to finish; do not
   create a replacement;
3. external baselines / paper analyses continue independently;
4. do not reopen marked-SMILES representation work before the main A7 result.
