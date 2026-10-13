# PR83 — append-only execution ledger

**Owner:** executing Codex / local run operator. **Scientific authority:**
`docs/CODEX_GOAL_PR83.md`. This ledger starts as an honest plan,
not an assertion that the experimental stages have been run. Do not replace
historical failed tasks. Add dated records with exact commands, attempt IDs,
checkpoint/input/code SHAs and artifact locations; corrections get a new entry.

## Source status at PR83 handoff

- PR83 stacked on open PR82.
- Current initial base commit: `ed52e9c123bcb60645596e9530f7aae60820c56e`.
  Later execution-related commits must be recorded separately.
- Existing GitHub Actions passed on the initial base: contract tests,
  documentation, existing agent-runtime tests, and evidence-layer tests.
  **These are CPU/CI integration checks, not a GPU endpoint result.**
- Final Qwen3-0.6B adapter and CRN-JEPA PR18 model are documented in
  `docs/CODEX_GOAL_PR83.md`; their actual availability in the
  job filesystem has **not** been verified by this PR.
- No valid S02–S05 Qwen/Graph GPU experiment or RL optimizer result is recorded
  here as of the document's creation.

## Stage ledger

| Stage | State | Attempt/task ID | Dataset and model SHAs | Result/artifact | Next action |
|---|---|---|---|---|---|
| S00 environment + actual checkpoints | PENDING | — | — | — | Inspect host and verify SHA; do not invent a path |
| S01 contracts + resume tests | PENDING | — | — | — | Confirm current head CI and target-host install |
| S02 routing/replay/chemistry parity | PENDING | — | — | — | Run 8→16 real checkpoint inputs |
| S03 Qwen-vs-Graph/Qwen valid128 | PENDING | — | — | — | Same frozen IDs and scoring |
| S04 graph-head endpoint RL train64 | PENDING | — | — | — | Verify true endpoint-positive contrast |
| S05 RL-head vs frozen-head valid128 | PENDING | — | — | — | Paired rerun, hashes, runtime |
| S06 two corrections if negative | NOT_YET_APPLICABLE | — | — | — | Distinct train-only corrections and honest diagnostics |
| S07 final result and promotion decision | PENDING | — | — | — | Measured report or documented hard blocker |

`PENDING` is not failure. `RUNNING` must have an
actual allocated task/POD or active local process; queue placement is
`QUEUED`. `PASSED` requires a complete artifact with
independent row/hash audit; finite train loss or script import is not evidence.

## Per-attempt record template

### YYYY-MM-DDTHH:MM:SS±08:00 — SNN/attempt-N

- State: QUEUED / RUNNING / PASSED / RETRYABLE / NEGATIVE / HARD_STOP
- Code commit / diff: …
- Exact invocation / environment: …
- Input path, counts, SHA-256, split and filtering: …
- CRN checkpoint SHA / Qwen base+adapter SHA / RDKit version: …
- GPU and resource consumption: …
- Output manifest / JSONL location / complete rows / independent readback: …
- Quantitative result: …
- First exception/failure evidence, root cause and patch: …
- Continuation decision and next bounded attempt: …

## Final comparison (populate ONLY from completed runs)

| Policy | 128 identical valid IDs | Structural exact | Full exact | Executor accepted | Qwen route/import calls | Wall seconds / reaction |
|---|---:|---:|---:|---:|---:|---:|
| Original frozen Qwen Stage I | Pending | — | — | — | — | — |
| Qwen control + frozen CRN FLOW | Pending | — | — | — | — | — |
| Qwen control + graph endpoint RL | Pending | — | — | — | — | — |

Record useful and failed candidates equally. No fabricated results, omission
of invalid rows or attribution of local oracle-state Event@1 to product-start.

## 2026-10-12 — formal scope and implementation synchronization

- Ruling: the user's later full-evaluation/no-smoke instructions supersede the
  original 128-valid / 64-train execution gates. The formal campaign traverses
  all strict train/valid/test rows and keeps the 28,971 official test denominator.
- Ruling: the latest PR83 handoff pins Stage-I State-SFT weights SHA
  `fbd8db06094fd8029d4cb0cac38cbb88280a442d76f686258194723019b1a1bd`.
  Prior Stage-II runs use weights SHA
  `40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75`
  and remain
  a separate historical condition, not a Stage-I baseline.
- The completed Stage-II single-pass task
  `meteor_mechet_pr83_flower_valid128_singlepass_8a100_qy_20261012_01`, instance
  `8b1d81dca1250f4401a12e1df01d0b75`, reached 46/128 strict endpoints and
  115/128 terminal episodes. Frozen pure Qwen was 47/128; 41 both correct,
  5 hybrid-only and 6 Qwen-only. First calls were identical on all 128 IDs.
  The legacy baseline has no authenticated shard contract. Raw inputs/actions
  remain in `outputs/eval/pr83_crn_qwen_flow_valid128_stage2_singlepass_20261012/`.
- Old RL instance `8b1d806ea1250fde01a12ba551810913` is END/task_killed;
  no RL head or successful optimizer result is claimed for it.
- Integrated upstream PR83 through `f976e45` (atomic head/optimizer/RNG/cursor
  resume; never advance a root after CUDA OOM) with the single-generation
  IMPORT correction and distributed full-evaluation shards. Resolved duplicate
  `--resume` registration, which was caught by the CLI regression test.
- Added per-shard authenticated resume, private-tail preservation, State-I vs
  history prompt validation, full-mixture endpoint metrics and authenticated
  frozen-graph pre/post pairing. Related tests: 22 passed before campaign tests;
  4 formal sequencing tests also passed. These are code checks, not model results.
- Formal commands/configs are in `scripts/run_taiji_pr83_formal_*_a100.sh` and
  `configs/taiji/meteor_mechet_pr83_formal_{pre,rl,post}_*_20261012_01.json`.
  The supervisor's durable status lives outside Git under
  `outputs/submissions/pr83_formal_campaign_20261012/status.json`.
- Formal output roots:
  `outputs/eval/pr83_formal_stage1_20261012/` and
  `outputs/agent/pr83_graph_endpoint_rl_stage1_train257167_20261012/`.
  No formal metrics are filled until the complete output audit passes.

### Formal allocation attempts

- Initial main Qingyuan task:
  `meteor_mechet_pr83_formal_pre_8a100_qy_20261012_01`, instance
  `8b1d822ca1250f4c01a12eae144c0ca5`. It remained resource-waiting despite
  remaining application quota. Cancelled before a GPU POD/output was created;
  END/task_killed verified. No unrelated task was stopped.
- Replacement ordinary AILab Qingyuan task:
  `meteor_mechet_pr83_formal_pre_8a100_ailab_20261012_02`, instance
  `8b1d822ca1250f4c01a12eb27a1d0ca8`. Same Stage-I checkpoint, full IDs,
  scripts and output root. Tencent taiji6, proven private Ceph init, 60-second
  default-POD heartbeat. At the latest check: PENDING/MPIJob created,
  application quota 50 with 8 applying, no assigned host/inference output yet.
- Sequential supervisor PID `2361335` records current status in
  `outputs/submissions/pr83_formal_campaign_20261012/status-ailab-attempt2.json`.
  The cancelled supervisor's status redirects there. Full train257167 RL and
  post-training full evaluation are prepared and will submit only after their
  preceding task succeeds and its complete report passes readback.
- Source SHAs verified locally: train
  `edc80c5c5eb13d50753c5566c5d6ac1b90b955b1a8dfece915241b2da4a40a75`,
  strict test `70a06f01a08490d4b5da602056a8f8e2f978128d3da3224d1601186529aeee3f`,
  official test `a0700444e03cdc73ef041ac1b33251486529976246e469c27e15f2bf6f296048`.
- Local final related suite: 26 passed. CI on `846cbca`: 4/4 passed.
  Queue admission and code checks do not establish a formal model result.

## 2026-10-12 — latest PR83 theoretical audit and bounded correction

- Upstream audited revision: `584bee6e8d93b53cb2a45adc53beff52c1634809`.
  This introduces Graph-EARHO and explicitly demotes the old fixed-GT-tail
  graph trainer to an ablation. The local formal launcher previously still
  selected that old script: entry/head/report contracts are now corrected.
- Confirmed missing `successor_group_advantages` import in the new main trainer;
  reproduced the actual unbound call. Combined with the stacked CLI it also
  duplicated `--source-sha256`. Both are now covered by executable regressions.
- Preserved single-generation greedy IMPORT/FLOW routing; upstream's old
  prefix-NLL router must not silently replace this already-repaired protocol.
- Corrected main actor support to terminal endpoints only and a matched K1
  follower. Local GT labels are separate replay supervision; the old mixed
  successor reward remains a named ablation. Horizon promotion now explicitly
  receives counts of endpoint-positive groups. Wrong observed terminals do
  not escape classification merely because the first successor matched GT.
- New prepared paths:
  `outputs/agent/pr83_graph_earho_stage1_train257167_20261012/` and
  `outputs/eval/pr83_formal_graph_earho_stage1_20261012/`.
  No historical artifact was overwritten, and no GPU job was submitted here.
- Full local code suite: **633 passed, 8 skipped** under the pinned
  RDKit-2026.03.4 wheel. The actual composed training CLI `--help`, launcher
  Bash syntax and `git diff --check` passed. None is an experimental result.
- A tensor/autograd counterexample shows local-successor reward can reduce
  true endpoint probability; another shows sampled conditional KL=0 does not
  constrain mass outside the slate. See `docs/GRAPH_EARHO_METHOD.md` and
  `tests/test_graph_earho_objective_boundaries.py`.
- Remaining mathematical boundary: Gumbel without-replacement Top32 selection
  plus success gating/partially sampled successor-class mass is a biased
  surrogate, not exact PPO/GRPO or an unbiased full endpoint gradient. A
  normalized-slate estimator is recommended but **not implemented** by this
  bounded repair. There is no convergence or endpoint-gain claim.
- Independent read-only review identified a reward-class error: map-free equal
  mixtures can project different structural precursors due to product-origin
  maps. A real RDKit example reproduced this. Pooling now retains the private
  mapped chemical state, bond form and isomeric serialization; continuation
  dedup also retains counters and cycle/history facts. Added origin and stereo
  regressions. Censor audits now follow the endpoint actor's semantics.
- Final full suite after the origin/stereo fix: **635 passed, 8 skipped**;
  independent read-only re-review found no remaining blocker introduced by
  these bounded changes. The broader estimator bias remains documented.

## Approved conditional-endpoint implementation plan (2026-10-12)

Authority: user approved the in-chat five-point repair and said to execute it.
Work remains in `MechET-pr83-greedy`; no job, push or merge is requested here.

1. Define normalized fixed-support q, IID replacement sampling, leave-one-out
   baseline and terminal-return score-function loss; verify exact expected
   gradient by enumerating a finite categorical example.
2. Add native source/sink/CONTINUE/COMMIT exploration proposals and merge them
   into the frozen support before normalization. Preserve original action and
   executor schemas; never inject GT as an on-policy draw.
3. Connect the trainer: deduplicate execution only, preserve draw multiplicity,
   freeze the K1 follower within an update, reserve short horizon for separate
   replay, and retain independently scheduled full product-start trajectories.
4. Update immutable reports/launch contracts and method derivation; run the
   complete unit suite and an independent read-only review, no model smoke.

Pre-flight: the new loss consumes the complete fixed pool, not just sampled
class logps; sampled indices refer to that same pool throughout execution.
Ruling: exploration expands the pre-update support, then IID sampling uses
the normalized expanded support — this gives a precise conditional objective
without pretending finite Top32 search covers the full action space. Cost if
wrong: missing-support actions still require explicit future exploration.
Ruling: endpoint actor updates are scheduled before observing branch returns;
all short-horizon groups remain replay-only, avoiding survivor-selection bias.
Cost: fewer terminal RL updates per root, rather than fabricated local credit.

## 2026-10-12 — approved conditional-endpoint plan implemented locally

- Task 1: complete-support normalization, IID replacement sampling and
  leave-one-out endpoint score-function loss implemented. Exhaustive finite
  categorical/autograd regression agrees with the exact conditional gradient;
  duplicate draws retain their original weight and short-H actor loss is zero.
- Task 2: native full-head source/sink/CONTINUE/COMMIT exploration implemented
  before support freezing. CPU tests use the actual external pointer decoder,
  verify summed event log probability and reject unfinished move-cap events
  rather than forcing COMMIT. No checkpoint or reaction experiment was run.
- Task 3: integrated full-support scoring before return observation, per-draw
  returns after origin/stereo-safe execution pooling, one fixed K1 follower per
  update, separate verified replay, whole-support frozen-original KL and
  independently scheduled full product-start autonomous trajectory capture.
- Ruling: trajectory journals store append-only evidence; checkpoints retain
  only journal offsets/hashes. Restore authenticates the saved prefix and
  quarantines later evidence before rollback. This avoids growing historical
  trajectory payloads in every checkpoint, while retaining crash evidence.
- Task 4: v4 launch/campaign seals and immutable configuration are prepared in
  `outputs/agent/pr83_graph_earho_conditional_stage1_train257167_20261012/`
  and `outputs/eval/pr83_formal_graph_earho_conditional_stage1_20261012/`.
  These replace the v3 prepared roots above, not historical output artifacts.
- Independent final read-only review found no Critical issue. Its Important
  precision-provenance finding was reproduced by a failing regression, then
  fixed: dtype and 4-bit mode now bind both resume and exported controller
  contracts, and the evaluation seal checks them. Default remains bf16 without
  4-bit loading. Stale censor/sampled-KL prose was corrected.
- Final verification after that fix: **651 passed, 8 skipped in 133.95s**, with
  pinned RDKit-2026.03.4 and external CRN source available for component tests.
  Training CLI `--help`, Python compilation, launcher Bash syntax and JSON
  contract parsing also passed. These are source checks, not endpoint results.
- All four implementation tasks are complete. Work is local in
  `MechET-pr83-greedy`, not pushed/merged; no new GPU job was submitted or
  historical task stopped. Main objective is conditional fixed-support
  endpoint REINFORCE plus explicit replay/KL, not full-product-start PG,
  PPO/GRPO, guaranteed candidate coverage or proven heldout improvement.

## 2026-10-13 — authorized diagnostic and joint-import handoff

- Explicit user exception authorized a small real-model test. Local T4/FP16
  validation8 baseline ended at 1 exact endpoint and 6 accepted terminals;
  22/22 imports and 34/36 FLOW events executed. Mean20.077s/reaction.
- Graph-only train32 finished traversing all32 roots but made zero updates:
  30 controller-blocked, 2 explicit-H alignment exceptions (maps33 and28),
  6 autonomous endpoint hits before any update. No final trained head/post-RL
  measurement exists. Source/result paths and concrete cases are recorded in
  `docs/results/PR83_JOINT_IMPORT_DIAGNOSTIC_20261013.md`.
- Two prerequisite bugs were reproduced RED and fixed GREEN: preserve H maps
  in alignment and remove the endpoint reward's extra unchanged-target veto.
  The primary scorer and legal executor are unchanged. Full suite after repair:
  **653 passed, 8 skipped in49.86s**. These are code checks, not a joint-model
  improvement or rerun of the failed diagnostic.
- User requested both code PR and training task for joint import–FLOW credit.
  Written architectural design/plan are committed for review; joint Qwen-LoRA
  training is not implemented yet and no new Taiji task was submitted.
  The existing graph-only script must not be deployed under a joint-model name.
- Ordinary quota readback: mainQY A100112/59/4/24, ZJK H20136/127/0/10,
  AILabQY A10050/0/0/0, V100146/0/0/0 (quota/using/applying/waiting).
  The successful main-QY `:taiji6` donor instance was rechecked; no allocation
  or infrastructure compatibility is inferred from unused quota alone.
- Independent read-only prerequisite review found no Critical/Important issue;
  its integration run passed14/14. Nonidentity H mapping, symmetric-H rejection
  and explicit-H stereo were also checked independently. Minor follow-up:
  persist those broader fixtures and an executor-to-heldout-scoring parity
  regression rather than relying only on the current expression-level test.

## Archived upstream notes merged from PR83 (pre-v4, historical)

The following records describe earlier source-only v2 repairs, not the current
conditional v4 estimator or the proposed joint-import implementation.

## 2026-10-12 Graph-EARHO methodological correction (source only)

- **Observed problem:** the original PR83 trainer used randomly selected last-three reference FLOW states, fixed 8-step rollout and action-wise teacher-forced mean NLL. This was not our specified EARHO. It is retained strictly as a named fixed-reference-tail ablation.
- **New code:** `src/mechet/graph_earho.py`, `src/mechet/graph_endpoint_rl.py` (actual graph event likelihood, successor-class objective), `scripts/train_crn_graph_earho.py`; CPU contracts `tests/test_graph_earho_contract.py`.
- **Method:** product-only first executed divergence; first graph-correctable FLOW gate; conditional reference fallback separately logged for Qwen mistakes; canonical executed-successor equivalence; bounded gold-free continuation; H=2→8 training-only competence promotion; signed class-level gradient; verified reference replay fallback.
- **Evidence:** no new real GPU training or valid128 endpoint/throughput results yet. Current task status remains unrun; do not mark S04 passed from green source CI.

## 2026-10-12 Graph-EARHO source-level repair

- Corrected potential structural endpoint error from using reference product
  atom maps instead of the autonomous model's mapped product origin.
- Replaced gold suffix anchor with actor-owned first graph FLOW-error state;
  earlier Qwen route/import/finish mistakes are controller-limited, not
  relabelled as graph on-policy corrections.
- Bounded continuation no longer compares heterogeneous Qwen-name and
  Graph-move log-likelihoods across parents. Unknown successors after
  finite search are censored rather than labelled failed chemistry.
- Reference replay is map-safe only when graph isomorphism gives a unique
  FLOW label. Original frozen CRN sampled-slate KL monitors head drift.
- Source-level tests and CI are NOT GPU-run evidence; no endpoint gain is
  claimed as of this record.
