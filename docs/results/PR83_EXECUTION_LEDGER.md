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
