# CODEX GOAL — PR #83: ship and measure CRN-JEPA + frozen Qwen + graph-only endpoint RL

## Latest user direction — 2026-10-13

The user requested both a code PR and a new Taiji task for **joint import–FLOW
endpoint credit**. This is a deliberate proposed change from frozen-Qwen
graph-only training. The written design and native implementation plan are
`docs/superpowers/specs/2026-10-13-joint-import-flow-endpoint-design.md` and
`docs/superpowers/plans/2026-10-13-joint-import-flow-endpoint-plan.md`.
They await architecture review; joint training is not implemented or submitted.
Do not launch the existing graph-only script and describe it as this new method.
This is an ongoing handoff, not a DONE/blocked terminal declaration.

An explicitly authorized T4/FP16 diagnostic finished: validation8 baseline
1/8 exact endpoints; graph-only train32 processed all roots with zero updates
(30 controller gates, two explicit-H alignment exceptions). The explicit-H
alignment and reward/evaluator discrepancy are now source-level prerequisites
fixed with regressions, not an empirical joint-model improvement. Preserve the
original artifacts and use the latest ACTIVE status/ledger for real task state.
Historical small-job instructions below do not authorize further smoke jobs.

## Current user override — 2026-10-12

The user requires full-scale evaluation and forbids new smoke, pilot, or small
subset jobs unless explicitly requested. This overrides the historical 8/16/128
validation and 64-root training steps below. Those sections retain the original
PR handoff for lineage; they no longer authorize small jobs.

The formal campaign uses the exact Stage-I adapter specified in this document,
without compact history. Run pure Qwen and frozen Graph-Qwen on all 2,890 valid
and 28,967 strict test reactions; retain the four unsupported official test rows
in the 28,971 denominator. Traverse all 257,167 strict training reactions for
graph-head-only endpoint optimization, recording reference replay failures and
unsupported/no-FLOW rows. Re-evaluate the trained head on the same complete
validation/test IDs and compare against the frozen graph control. Training
rewards, learning rate and horizon are frozen before accessing test scores.

Use `scripts/run_pr83_formal_campaign.py` for sequential formal submission:
one eight-A100 host for independent evaluation shards, then one A100 for graph
head training, then the eight-worker post-training evaluation. At most one
campaign task is active. The smaller allocation applies to the single-policy
trainer; eight-GPU evaluation runs eight independent policy workers. Every job
uses the proven private Ceph initialization and default POD heartbeat logs.

Historical Stage-II validation outputs are preserved as a separate condition;
they do not satisfy the Stage-I formal comparison. The completed single-pass
Stage-II run recorded 46/128 endpoints versus 47/128 pure Qwen and identical
first calls on all 128 IDs. No endpoint gain is established by that result.

The original 64-root RL task was stopped at the user's request. It remains
stopped; the formal campaign creates a distinct train257167 attempt.

**Status: EXECUTE, not another proposal.** This document is the handoff for the
existing stacked MechET code PR #83, branch
`feature/crn-jepa-qwen-electron-routing-20261011`, based on PR #82.
The user explicitly requires **persistent, result-oriented execution**. A
successful CI run, a failed first smoke test, a zero-endpoint pilot, a Qwen
routing mismatch, and an initial CUDA/RDKit compatibility error are **not**
completion conditions. Continue with the next concrete repair or diagnosis
while observing the non-negotiable integrity/resource bounds below.

**Read first**: `AGENTS.md`,
`PROJECT_MEMORY.md`,
`docs/ACTIVE_ICLR_STATUS.md`,
`docs/CRN_QWEN_GRAPH_ENDPOINT_RL.md`.
Check both PR heads and the files actually available on the compute host.
Do not use stale PR status, claimed completed jobs, or invented checkpoints.

## Absolute objective

Run a **real product-only 128-reaction paired experiment** using:
1. The existing frozen **Qwen3-0.6B State-SFT final adapter**, still responsible
   for IMPORT / FINISH and for tool selection. This is not a retrain.
2. The **actual verified CRN-JEPA PR18 graph checkpoint** plus its existing
   source/sink/commit decoder, solely for FLOW arguments.
3. The **unchanged MechET executor**, which alone defines next states and
   endpoints.
4. **Graph-EARHO**: autonomous first-divergence diagnosis, chemical
   successor-equivalence pooling, adaptive forward correction horizon from first model-owned FLOW error and
   train-only verified reward. The old last-three-reference FLOW group-RL
   trainer is a **named ablation, not the main method**. Use the exact same 128-ID
   before/after comparison.

**Non-negotiable outcome**: produce independently readable per-reaction
JSONL, a denominator/ID completeness audit, exact structural endpoint Top-1,
full-mixture exact (separate), rejected/unsupported/unchanged-target counts,
first divergence cause, inference time with Qwen/graph/executor breakdown,
effective RL groups, and hash-pinned artifacts. **Do not report gains before
they are measured.**

## Fixed inputs and evidence

- MechET PR82 Stage-I final Qwen adapter directory:
  `outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17`,
  weights SHA-256
  `fbd8db06094fd8029d4cb0cac38cbb88280a442d76f686258194723019b1a1bd`.
  The wrapper task failed AFTER saving the weights; independently verify
  artifact/manifest before loading. Do not treat that wrapper as successful.
- Qwen base: `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`.
  There is no confirmed 0.8B PR82 adapter; do not relabel 0.6B as 0.8B.
- CRN-JEPA PR18 checkpoint in the **separate** CRN-JEPA checkout:
  `outputs/auto_research/pr18/full_corpus_pretrain_v1/train_full_8a100_v2/latest.pt`,
  SHA-256
  `bb2f7fe53e87fc0e2438c27065ceb588dcd38e604959c9d69fb3984885a376ee`.
  The 82.16% inverse Event@1 is on oracle current states, not product-start.
- MechET strict executable split: train 257,167 / valid 2,890 / test 28,967.
  FlowER complete endpoint test: 28,971. Never mix them or read locked test
  in this development campaign.
- One identical 128-ID validation cohort, selection seed 17, one K=1 episode
  per reaction, 40 decision / 32 fragment-import budget, no value critic.
  Use the *same recorded IDs* and the same executor for every comparison.
- Known 112/2,890 aromatic / Kekulé private-remap discrepancies are a
  **chemical-parity audit**; do not silently exclude cases.

## Execution stages — do them, in order

| Stage | Work | Required evidence; no premature STOP |
|---|---|---|
| S00 | Confirm two local checkouts, actual SHA-pinned weights, Qwen adapter manifest, RDKit 2026.03.4, available GPU and data shards. Search legitimate pre-existing mounts, not just the first hard-coded path. | Emit paths, SHA, GPU name, runtime versions and exact 128 IDs; if a checkpoint path is stale, locate it and revalidate its SHA. |
| S01 | Run `python -m compileall`, focused tests and documentation check. Add tests for failed first decision, resumed JSONL, import-prefix parse, multi-move schema/BE_DELTA and scorer parity. | Pass in CI and real target environment. On error: read traceback, fix implementation, rerun. Green CPU CI is not the destination. |
| S02 | Real weights: 8 then 16 valid product-start rows and state-level FLOW import/execute parity. Verify Qwen tool-name router *against* the frozen original Qwen tool-selection path on the same observations. | Record valid/invalid graph events, correct tool decisions, fragmented import behavior, first chemistry divergence and Kekulé cases. If parity fails, **fix and retest** before expanding; retain original Qwen baseline. |
| S03 | Run matched original-Qwen and frozen Graph-Qwen on **all the same 128 valid IDs**. No reference state in either actor. | Per-reaction outputs; endpoint Top-1, failures by stage, actual wall time per reaction and number of Qwen calls. Use `--resume` for interrupted PR83 evaluation, never delete partial output. |
| S04 | Execute **Graph-EARHO**: 8-row real-weight smoke, then 64 train-only roots with first *actual executed* graph-correctable divergence; group candidate actions by chemical successor; adaptive H=2→8; K=8 and bounded continuation. Old fixed-reference-tail RL is a separately identified ablation. | Show actual effective positive+negative RL groups, full endpoint-positive counts, reference-only fallback count, train steps/gradients and checkpoint hash. A nonzero SFT loss or saved file alone does not count as RL. |
| S05 | Evaluate the trained graph head on **the exact same S03 128 IDs** and compare to frozen Graph-Qwen; original-Qwen reference remains unchanged. | Paired successes/losses, CI or exact discordant cases, seconds/reaction, graph validity, no-op finishes, import and routing decomposition. |
| S06 | If a result is zero/negative, run **at least two targeted, resource-bounded diagnosis/correction cycles** before concluding. Recover from specific observed errors; do not launch another general research program. Repeat S02/S03/S04/S05 where relevant. | Each cycle includes a falsifiable cause, code/test patch, new run ID/artifact, and same 128 IDs. Do not reinterpret a negative scientific result as an infrastructure error. |
| S07 | Produce a final PR-facing empirical report, failure table, reproducibility commands and explicit merge recommendation. Expand to 2,890 validation only if the measured 128-row evidence justifies it. | Commit authentic numeric output/manifests where appropriate; keep large private raw artifacts in the documented storage and preserve their SHAs. |

### Specific fixes to prioritize (do not spend days theorizing)

0. **Do not downgrade our RL method:** Read `docs/GRAPH_EARHO_METHOD.md` and run `scripts/train_crn_graph_earho.py` as the manuscript main method. Preserve `scripts/train_crn_graph_endpoint_rl.py` as fixed-reference-tail ablation. If first deviation belongs to Qwen IMPORT/FINISH, report it as controller-blocked, not a Graph-EARHO on-policy frontier. An all-negative successor class group gives no policy gradient; use exact source/sink/commit log-probability and state-equivalence marginalization.
1. **Qwen tool-route parity**: short tool-name mean-NLL is NOT the same as
   Qwen's original generative tool selection. Compare identical state prompts.
   If discrepant, implement a minimal partial tool-name generation route that
   does not generate FLOW arguments, and run a router ablation. Label the route
   change independently from the graph policy; do not disguise it as fixed.
2. **FLOW graph/executor adapter**: explicitly exercise the 2-electron LP/BOND/
   ATOM pathways, coupled moves, explicit H, aromatic/Kekulé states, map
   renumbering, and every produced FLOW against
   `verify_electron_step`. BE_DELTA is not covered by CRN FLOW v1.
   Do not send it secretly back to Qwen or exclude failures.
3. **Actual speed**: CRN-JEPA's existing `decode_events` contains
   Python tensor-to-scalar sync and Python beam loops. Profile GPU prefill,
   graph forward, beam decode, RDKit execution and import generation; batch
   pointer scoring rather than assuming graph inference is fast. Keep numerical
   event parity before making kernel changes.
4. **RL signal**: keep all-negative groups zero advantage. Report how many
   positives are **complete verified endpoints**, not only reference-successor
   matches; if zero, test one short late-frontier curriculum correction and one
   matched longer bounded continuation, under distinct train-only configs.
   Do not claim endpoint RL from a reference-only run.
5. **Recovery**: existing hybrid evaluator has a SHA-locked `--resume`
   path; preserve its completed IDs and partial tail. Use the equivalent SHA-locked `--resume` training checkpoint path, which
   persists model-head parameters, optimizer state, RNG, cumulative group audit
   and the next training-root index. Never replace raw failed attempts.


The graph RL trainer now persists `latest_train_state.pt` every
4 processed roots by default (`--save-every 4`). For a transient
interruption, reissue the **same** train-only invocation with `--resume`
and exactly the same code and input SHA; if code or scientific settings change,
use a new immutable output directory. Never bypass hash checks to force a
resume.

## What Codex must do after problems arise

| Observable problem | Required next action (unless hard stop) |
|---|---|
| Python import, dependency, tokenizer, `logits_to_keep`, or shape error | Minimize to reproducible case; patch the adapter; add regression test; retry exact stage. Check installed model API rather than assuming. |
| Missing file/checkpoint path | Inspect verified shared mounts, documented task outputs and SHA manifests. Try a compatible user-authorized mount; do NOT fabricate/retrain the missing model. If unavailable, still finish static, CPU and fixture work and record precise external blocker. |
| RDKit mismatch or rejected electron action | Reproduce on a named frozen input, compare parsed atom maps and Kekulé state, patch only legitimate runtime representation errors, audit parity. No silent reference-based correction. |
| CUDA OOM | Reduce independent proposal microbatch and scheduling concurrency first. Do not change held-out IDs, candidate budget, trained weights or chemistry semantics. Retry at most three compatible configurations. |
| Taiji resource queue/temporary network issue | Record status; choose a supported available existing A100/H20 queue when permitted; resume same stage with a different run ID, no redundant simultaneous expensive jobs. Inspect logs/POD/artifacts before claiming success. |
| One failed rollout, empty candidate list or malformed fragment | Count failure with its original ID, continue other IDs; preserve denominator; later aggregate by root cause. |
| Zero endpoint hits or negative 128-result | **Not a stopping condition.** Inspect first divergence and sample supports, run two bounded causal repairs and matched reruns. A negative after those is a completed **negative result**, not permission to invent performance. |
| No effective RL groups | Run one reference-supported correction of training-frontier selection and one budget-matched alternative rollout horizon, separately versioned, on train only. If still none, report honest reference-SFT-only evidence and complete the non-RL hybrid comparison. |

## Retry / resource policy

The task is not open-ended self-research. It is a finite executable campaign.
Use existing 1×A100 or 1×H20 when available; avoid 8-GPU allocations for one
0.6B Qwen plus graph model. Use the same frozen dataset/checkpoint/metric
contracts. Up to **3 environment-compatible recovery attempts per failed stage**
are permitted, and at most **2 scientifically distinct corrective cycles**
after a complete negative S03 or S05. Do not rerun expensive completed work.
Every retry writes a **new immutable attempt ID** and points to preserved
previous artifacts. If extra compute or a new scientific hypothesis is needed,
record that as a proposal, not a silently applied modification.

Make forward progress without requesting approval for routine bug fixes,
tests, local code organization and authorized bounded retries. Do not request
user decisions that are already settled in this file.

**Hard stops only**: unsatisfied source/provenance hash, test/reference leakage,
corrupted benchmark boundary, impossible chemical execution under the locked
contract after audited reproduction, no authorized access to essential
private weights/data/GPU after enumerated recovery attempts, or exceeding the
explicit resource bound. A hard stop must include failed commands, traced
cause, exact missing artifact and the next unblock action — never just
"can't run" or "no signal".

## Authoritative deliverables (must appear before declaring DONE)

- Update `docs/results/PR83_EXECUTION_LEDGER.md` with real stage
  states, task IDs, hashes, selected IDs, paths and negative attempts.
- Persist `results.shard-00-of-01.jsonl` plus `summary.json`
  for each complete 128-row product-only comparison; use SHA-bound evaluator
  manifests. Do not overwrite a previous run.
- Train report: `graph_correctable_frontiers`, `controller_blocked_frontiers`, `reference_conditional_frontiers`, `pooled_duplicate_actions`, `curriculum_promotions`, `effective_rl_groups`,
  `endpoint_success_candidates`,
  `reference_successor_candidates`,
  optimizer steps, frozen Qwen SHA and RL-head SHA.
- Table of identical 128 IDs: original Qwen vs CRN/Qwen frozen vs CRN/Qwen
  graph-head RL, structural Top-1 numerator/denominator, pairwise delta,
  execution validity and time.
- CI green **and** the factual GPU-state report; document any exact source
  contract remaining unverified.
- Update PR #83 description and commits with final actual findings, marking
  unmeasured analyses explicitly. Do not merge automatically.

Allowed terminal states: `DONE_MEASURED_GAIN`,
`DONE_MEASURED_NEGATIVE` (only after two bounded corrections),
or `BLOCKED_WITH_EVIDENCE` (only after exhausting permitted
recovery or a hard integrity stop). **"CI passed", "started a task",
"tiny smoke ran", "RL loss finite", or "zero wins on first try" are forbidden
terminal states.**

### Graph-EARHO correction verification (2026-10-12)

Do not select a gold near-end anchor in place of the model's first graph
error. Validate `tests/test_graph_earho_runtime_contract.py`, reference
atom-map correspondence, and model-created product-origin endpoint projection.
Mixed Qwen name-token scores and graph source/sink scores are incomparable
across beam parents. Censored legal successors receive zero explicit negative
advantage. Report frozen CRN sampled-slate KL, replay-only updates, and
graph-correctable vs controller-blocked populations. CUDA parity and measured
product-start results are still required.
