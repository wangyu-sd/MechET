# Codex execution handoff — Route-RL correctness × effective-gradient × GPU efficiency
**Date:** 2026-11-11  
**Repository:** MechET, new Draft PR based on `main`  
**Role:** implementation/audit/engineering optimization handoff to Codex, not a new learning-method proposal  
**Status:** the first Route-RL batch below was measured on 2026-11-06; no new GPU job or route-model evaluation is claimed by this PR.

## Objective and strict boundaries

Make the **existing** target-only, multi-reaction Route Dr.GRPO pipeline more efficient and auditable, in this exact order:

1. **P0: Verify reward/correctness labels and recoverable training signal.** Show what generated trajectories are actually counted as valid/closed/reference-supported, identify erroneous successes/negatives, and measure exactly how much learning signal is obtained per unit of work.
2. **P1: Profile end-to-end collection and learner time.** Disentangle GPU prefill/decode/learner compute, CPU electron execution, reward scoring, I/O, cross-rank synchronization and warmup.
3. **P2: Fix only measured engineering bottlenecks** (batching, bounded ready queues, cached deterministic encoding/prefill if exact, batched durable writes, excess kernel autotune / filesystem metadata operations).
4. **P3: Evaluate optional conditional-branch overhead** against the existing root-G4 campaign, **without enabling PR #103 by default** or asserting it is an improvement.

**No scope expansion:** do not replace or retrain the Stage-II model, alter `IMPORT/FLOW/FINISH` or electron executor, modify hidden reference reward coefficients/logic, introduce physics/DFT/forward critics, change ReFIT, launch a new optimizer search, or filter/select easier targets. If a reward/verifier bug is found, provide a minimal failing replay/patch with an isolated experiment condition; keep the frozen existing baseline intact for paired comparisons.

## Source-of-truth and first code-discovery task

- Official Route-RL design/protocol: [2026-11-05 target-only Route-RL design](https://github.com/wangyu-sd/MechET/blob/docs/target-only-route-rl-design-20261105/docs/design/2026-11-05-target-only-route-agentic-rl.md), especially ` 4–6, 8–13. Snapshot documented at **2026-11-06 16:24 CST**.
- Optional conditional-fork experiment (not integrated): [PR #103](https://github.com/wangyu-sd/MechET/pull/103), branch `feat/route-state-conditional-credit-20261111`, `src/mechet/route_branch_credit.py`.
- Current repository main contains `src/mechet/trace_agent_env.py`, `src/mechet/endpoint_process_rollout.py` and `scripts/train_endpoint_process_rlvr.py`, but these are **not** the final external multi-reaction route trainer.
- On inspected Git branches, there is **no committed** implementation under the design's `TargetOnlyRouteEpisode` / `route_policy_sampling.py` / `optimize_routes` names. The design reports that a real 4×H20 job completed an optimizer update, so its actual runnable code/checkpoint may live only on a shared working tree / submission package. **Find and audit the real code before editing. Never pretend the older per-event RLVR scripts are the route trainer.**

Initial read-only discovery on a machine with shared storage:

```bash
export ROOT=/aaa/fionafyang/buddy1/whaleywang/MechET
export DATA=/aaa/fionafyang/buddy1/whaleywang/datasets
cd "$ROOT"
pwd
git status --short
git rev-parse HEAD
git branch --show-current
rg -n "TargetOnlyRouteEpisode|HiddenRouteReward|optimize_routes|route_policy_sampling|external_stage2_target_only_route_dr_grpo" src scripts docs 2>/dev/null | head -n 100
ls -l outputs/submissions/external_route_stage2_dr_grpo_20261106_03/first_optimizer_update_witness.json
ls -ld outputs/agent/external_stage2_target_only_route_dr_grpo_v2_20261106_03/checkpoints/round-000001
```

If the training code is only inside a submitted artifact or shared/private deployment, inspect its **actual source SHA** and record its runtime path, hashes, command, model/adapter, output and recovery contract. Port minimal reproducible fixes into this PR, rather than inventing a second, divergent learner. If access is unavailable, provide cross-machine commands with the exact known absolute paths and distinguish unverified code assumptions from observed results.

## Frozen reference run: first collected batch (not steady-state speed)

Machine-readable snapshot: [`configs/ops/route_rl_first_update_snapshot_20261106.json`](../../configs/ops/route_rl_first_update_snapshot_20261106.json).

- Full intended official condition: **TRAIN 23,730 / VALID 2,986**, **G=4 independent target-only rollouts**, four H20 GPUs, Stage-II History-SFT checkpoint-89442, fixed route reward, one target exposure and 371 intended scheduled batches.
- Measured first batch: **64 target groups / 256 trajectories / 5,030 actor decisions / 622 completed reaction programs**.
- **74/256** structurally stock-closed, only **9/256** also supported by hidden reference; **12/64** groups supplied nonzero *group-relative* advantage; **0** resource/protection-censored groups.
- Terminal categories in the design summary: 92 action-rejection stops, 77 cycle stops, 13 nonproductive-reaction stops; use actual raw records to verify mutual exclusivity and denominators.
- Cold first-round sample+reward+cross-rank wall clock ≈ **2,888 s (48.1 min)**. Includes startup/autotune/synchronization; **not** a measured steady-state sampling throughput.
- Process snapshots saw some Triton autotune waits and shared Ceph `exists/makedirs` metadata waits, with uneven GPU utilization. The relative time fractions have **not** been measured.

**Do not report “81.25% wasted tokens”:** 52/64 zero-relative-advantage groups is a **group-level PG observation**, not a token fraction or proof that KL updates used zero compute. Measure real assistant tokens and learner gradient contribution before asserting savings.

## P0 — Correctness and training-signal audit (read-only first)

### A. Trajectory-level labels from the same saved executions

Build an **offline, read-only audit** over the first complete batch (and subsequently all existing available rounds) using original IDs, behavior-policy hashes, frozen stock and evaluator receipts. Per target and trajectory emit:

- `target_id, trajectory_id, group_id, behavior_version, actor_sha, reward_schema_version, stock_sha`.
- Original model calls and assistant generated tokens, accepted/rejected FLOW, IMPORT, FINISH count, completed reaction occurrences, executor duration, termination reason, censor class.
- `execution_valid, root_connected, acyclic, all_occurrences_closed, structural_stock_closed, reference_supported_closed, unsupported_closed, trusted_root_connected_edges, partial_progress, reward_value`.
- `auxiliary_fragments` and participant provenance/status **separately**: do not blindly require every reagent/ion/catalyst to be stock and do not quietly discard an indispensable consumed nonstructural component.
- Any invalid/undecidable classification must include exact source record and a reason; distinguish chemical conflict from unknown reference coverage.

**Important:** `reference_supported_closed` is a benchmark-specific supervision label, **not** proof of all possible chemical correctness. `structural_stock_closed` is computational closure, **not** laboratory feasibility. Do not relabel all unsupported-closed proposals as wrong, and do not give them a newly invented full reward. Reference routes remain fully hidden from actor sampling and only accessible after termination.

Require invariant tests/replays for: repeated precursor occurrences (not set-collapsed), full AND closure, non-stock auxiliary roles, charge/stereo identity, connected support propagation, rejected action handling, invalid/empty FINISH, nonproductive/cycle routes, unsupported stock-closed, resource/protection censor. Compare offline reconstructed reward against the recorded scalar reward **bit-for-bit or within justified numeric tolerance**. Every mismatch must identify the raw artifact and fail validation. No reward changes in this audit PR by default.

### B. Group-level effective-signal ledger

For each frozen root target group, record `G, rewards[], terminal_statuses[], usable, R_mean, R_std, A[], nonzero_advantage_count, all_equal_rewards, generated_tokens, model_calls, learner_policy_tokens, KL_tokens`. Report:

- **Informative group fraction:** groups with nonzero conditional/root centered advantages divided by ALL scheduled target groups, with usable/censored denominators separately.
- **Effective-group yield per GPU-hour** and per **1 million charged generated assistant tokens**, with zero-advantage groups included in the work denominator.
- Fraction of actual assistant policy tokens belonging to groups with nonzero PG advantage; distinguish **nonzero advantage** from **nonzero effective clipped gradient**.
- Reference-supported completion, unsupported stock-closure and partial reward histograms; count how many rewards saturate at 0.2 while remaining obligations are unresolved.
- Invalid actions within eventual positively rewarded trajectories, collapsed duplicate successors, zero-variance groups, censoring by route depth, AND count and cohort.
- NO use of TEST, external 460, SimpRetro or held-out validation labels for selecting training objectives. Frozen FULL-VALID2,986 remains the model selection boundary.

Do not re-label original `G=4` independent root samples as the `G=2` conditional sibling samples of PR #103.

## P1 — True performance profiling with phase timestamps

Instrument the **actual target-only multi-reaction trainer** (after locating it) with low-overhead rank-local monotonic timestamps and batched event records. Capture:

- GPU: model load and Triton/autotune warmup; tokenizer/chat-template rendering; prompt encoding; prefill vs decode (where engine exposes this), sampling model requests, sampled tokens, KV/cache hit if available; learner forward/backward/gradient reduction/optimizer step.
- CPU: executor queue wait, execution, structural precursor extraction, route graph/certificate bookkeeping, hidden scorer and reward graph lookup.
- Scheduling: ready/in-flight/blocked queue lengths, prompt-length buckets, batch padding/utilization, stragglers, empty-rank barrier time and cross-rank synchronization.
- Storage: call-count and elapsed time of file existence checks, directory creation, read/write/fsync, checkpoint serialization and shared Ceph metadata operations. **Do not include source molecule structures or private reward annotations in distributed actor-visible telemetry.**

Keep clocks within rank and monotonic for durations; do not add independent per-call Ceph files. Use batched durable local scratch writes / flushed structured logs on shared storage, and preserve immutable source receipts. Label cold start separately from steady-state; collect enough later completed batches to report an observed steady-state distribution, rather than extrapolate from batch 1.

Output a machine-readable `phase_profile.json` and a per-rank per-phase table: count, total time, median/p95, sampled tokens, model calls, queue stalls, GPU busy/idle proxy, peak memory and uncertainty/coverage. Reconcile summed phase times with actual wall clock, accounting for overlap; **do not sum overlapping CPU/GPU durations into wall time**.

## P2 — Only after profiling, fix highest-confidence bottlenecks

In order of minimal intrusion:

1. **Ceph metadata storms:** cache invariant paths/manifest IDs and deterministic encodings; batch log receipts; avoid redundant `exists/makedirs/fsync` per action where correctness permits; do not weaken atomic checkpoints or replay provenance.
2. **GPU/CPU overlap:** maintain bounded asynchronous ready queue / executor worker pool and length-aware generation microbatches; preserve **one frozen behavior version per collection**, original action IDs and trajectory independence. Do not use a globally mutable model while sampling.
3. **Warmup/prefill:** perform safe one-time kernel warmup outside measured steady-state window. Profile true prefix equality and cache eligibility before attempting KV reuse; only identical tokenized prefixes qualify. Model, tokenizer, prompt, tool schema and captured logprobs must remain identical.
4. **Learner compute:** reuse exact deterministic token/logprob computations when provably same; preserve `Dr.GRPO` fixed global `N×G×C` denominator, assistant-only token masks, clipped positive/negative advantage, true behavior/reference logprobs, KL derivative and all-reduce semantics. Optimize memory/chunking without altering gradients.

Avoid full async cross-policy-version RL, speculative decoding, new vLLM/SGLang backend requirements or speculative data/reward balancing until a measured bottleneck justifies their extra compatibility work. If a backend is replaced, prove model/tokenizer/action/executor/logprob equivalence first.

## P3 — Optional PR #103 efficiency accounting (not a method rollout)

PR #103's `collect_conditional_group` currently loops over sibling continuations sequentially, and a restored executor snapshot is **not** a reused transformer KV cache. Audit its total root-prefix + suffix tokens/calls and extra executor cost. If the real route adapter exists, compare **root G4 / boundary conditional / pre-FINISH conditional** under matched work with unchanged hidden evaluator, and report informative group fraction and cost. Do not promote this method based on toy tests or average loss alone.

The **present PR has no dependency on merging PR #103**, and must not silently enable `conditional_electron_branch_v1.json`.

## Acceptance, evidence files and Codex completion contract

1. **First deliverable:** source-location/provenance audit plus offline first-batch reward & informative-group report using existing saved data. Explicitly label any missing private runtime files instead of fabricating numbers.
2. **Second deliverable:** real per-phase profiler in the actual trainer, tests for log output integrity, timing, token counts and no reference leakage; report initial bottleneck breakdown from measured rank-level files.
3. **Third deliverable:** smallest measured performance fix(es) with true A/B comparison using the **same frozen actor, target IDs, G, sampling seeds, prompt/tool schema, inventory and reward implementation**. Compare steady-state target groups/hour, GPU-hours per *informative* group, per-target/route reward outcomes and censoring, not just raw tokens/s.
4. **Regression gate:** replay exact original reward/certificate labels and reject unexpected training gradient, sampling distribution, executor or protected-target divergence. Nonzero PG group count can fluctuate with stochastic sampling; interpret paired and adequately repeated measurements, not a single anecdote.
5. **Final evidence:** a concise Results Markdown summarizing measured vs inferred effects, per-rank profiles, raw record location, code/actor SHA, runtime versions, test commands, observed speedup and uncertainty, and any unresolved correctness issues.

Shared output folder (new files under shared storage, not local machine):  
`/aaa/fionafyang/buddy1/whaleywang/datasets/mechet_route_rl_efficiency_audit_20261111`  
Never commit private raw route references, run artifacts, model weights or hidden reward indices.

When GPU access, quota, environment packages or the private route trainer is missing on the current machine: **do not shrink full scientific cohorts, repeatedly trigger OOM, or silently stop at synthetic tests.** Provide copy-paste commands with the absolute shared paths, environment setup, checkpoint, log/manifest, resume operation and post-run summary for the eligible GPU host. CPU contract tests are allowed to verify software correctness but must **never** be presented as chemical performance evidence.

**Stop conditions / no scope creep:** A failure to reproduce the recorded reward is a blocker for performance claims, not permission to redesign the reward. No new models, physics-based environments, forward chemical verifiers, curriculum, reference-guided rollouts, expanded tasks, unnecessary new PRs or unrelated research planning. Apply changes incrementally **in this PR**; keep it Draft until independently reviewed.

## Useful internal links

- [Route Dr.GRPO protocol and first-batch evidence](https://github.com/wangyu-sd/MechET/blob/docs/target-only-route-rl-design-20261105/docs/design/2026-11-05-target-only-route-agentic-rl.md)
- [Current PR #103 conditional branching, separate and off by default](https://github.com/wangyu-sd/MechET/pull/103)
- [Existing per-event RLVR rollout, *not* the route-level trainer](https://github.com/wangyu-sd/MechET/blob/main/src/mechet/endpoint_process_rollout.py)
- [Executed electronic-state search route certificate, *not* the hidden reward evaluator](https://github.com/wangyu-sd/MechET/blob/feat/flower08b-unified-electron-tree-20261103/src/mechet/reverse_route_network.py)
