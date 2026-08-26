# Active ICLR status

> This file is the short-lived execution authority for the current ICLR work.
> `PROJECT_MEMORY.md` remains authoritative for historical dataset/model lineage.
> If an older operational note in `PROJECT_MEMORY.md` conflicts with this file,
> this file wins for **current experiment priority and A7 observation choice**.

Last updated: 2026-08-26.

## What is already on `main`

- PR #35 (`b7e26027`) merged the full-data protocol/runtime update.
- PR #40 (`4b410e52`) merged the code/protocol repair for Issue #36:
  matched mech-USPTO training/inference prompt + tool schema + 12-call budget,
  one generation worker per 40-GiB A100, explicit failed/OOM candidate accounting,
  NLL/first-candidate metrics, and frozen lineage hashes.

The old protocol-mismatched ~1.60% mech-USPTO action-delta result remains
**diagnostic only**.

## Only two active ICLR issues

### #36 — matched mech-USPTO action-delta rerun

Implementation/protocol repair is complete on `main` via PR #40.
The issue stays open only until the complete K=10 + NLL-ranked rerun artifact
finishes and is validated. Do not create another implementation branch unless
the frozen rerun itself exposes a new bug.

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

## Consolidated / deferred issues

- #38 is superseded by #37. Its useful content (lossless removal of redundant
  state/proof serialization) is now part of the compact-full-state A7 plan.
- #39 reaction-site-marked SMILES is deferred beyond the current ICLR paper and
  must not block A7.

## Priority order

1. implement + launch #37 compact-full-state A7;
2. finish the already-prepared #36 matched rerun artifact;
3. external baselines / paper analyses continue independently;
4. do not reopen marked-SMILES representation work before the main A7 result.
