# Active ICLR status

> This file is the short-lived execution authority for the current ICLR work.
> `PROJECT_MEMORY.md` remains authoritative for historical dataset/model lineage.
> If an older operational note in `PROJECT_MEMORY.md` conflicts with this file,
> this file wins for **current experiment priority and A7 observation choice**.

Last updated: 2026-08-26.

## A8-Lite targeted continuation pilot (2026-09-03)

The first low-cost A8 learning pilot is initialized from the completed
compact-full-state A7 adapter at
`outputs/agent/tool_sft_flower_compact_full_state_qwen3_8b_a100_20260826`.
It trains one targeted epoch on 30,000 deterministic late-state reset
continuations plus 10,000 disjoint complete expert anchors sampled only from
the frozen FlowER train split.  This is a history-robust continuation pilot;
it is not off-policy recovery from arbitrary erroneous chemical states.

The active ordinary Qingyuan 8xA100 task is
`meteor_mechet_a8_lite_state_reset_qwen3_8b_1ep_8a100_qy_20260903_02`
(instance `8b1d80eba065f55b01a06652c10b00e8`).  It obtained a real POD, completed
distributed pretokenization, launched eight `train_tool_sft.py` ranks, and all
eight A100-SXM4-40GB GPUs were observed at 100% utilization.  The `_01` task
ended during startup because its isolated implementation worktree did not
contain the local wheel artifacts; `_02` resolves wheels from the established
main Ceph artifact directory.

Validation uses all 2,890 strict validation reactions.  Any eventual test
result must use all 28,967 strict executable test reactions with missing
predictions counted as failures.  Looped-layer execution, value learning and
off-policy recovery remain gated on this pilot and are not part of `_02`.

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
