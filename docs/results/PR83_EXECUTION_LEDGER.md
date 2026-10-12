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
