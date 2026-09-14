# Active ICLR status

> This file is the short-lived execution authority for the current ICLR work.
> `PROJECT_MEMORY.md` remains authoritative for historical dataset/model lineage.
> If an older operational note in `PROJECT_MEMORY.md` conflicts with this file,
> this file wins for **current experiment priority and A7 observation choice**.

Last updated: 2026-09-14.

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
