# Shared Electronic-Route MCTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline, task-by-task.

**Goal:** Implement and submit genuine shared MCTS for all460 R1/R2 targets, preserving the existing actor and completed greedy results.

**Architecture:** A planning node includes unresolved precursor obligations and an active electronic context. Shared sampled-action PUCT selects edges, real executor rollouts cross reaction boundaries only at finish_trace, and complete stock closure backs up1 along the simulated path. Existing generation/execution caches and native journals are reused.

**Tech Stack:** Python, existing RDKit/vLLM0.8.5 actor and executor, Ceph POSIX locks/checksummed JSON, pytest.

**Spec:** docs/superpowers/specs/2026-10-14-shared-electron-route-mcts-design.md

## Global Constraints

- Same actor SHA40204...7e9de75, full460 targets and cohort stocks; no GT in search.
- No extra model/value/training/RL; no fixed chemical depth or total branch cap.
- Keep old adaptive outputs immutable; only R1/R2 gets a replacement task.
- One ordinary8A100, Tencenttaiji6, proven private Ceph, meteor prefix,45s default stdout.
- No new model smoke/subset experiment; CPU deterministic tests are code verification only.

## Review Focus

- Shared transpositions must include stock, pending obligations, ancestry and exact electronic history.
- Only ALL stock leaves give positive return; finish/EOS/one stock leaf must not.
- Invalid samples return0 for that rollout; context/memory interruption is censored.
- Concurrent/crashed backups must not double-count or leave permanent reservations.
- Runner must visibly use MCTS and persist recoverable progress, rather than silently call the old selector.

## Task 1: Shared MCTS planning controller

**Files:** Create scripts/shared_route_mcts.py and tests/test_shared_route_mcts.py.

**Interfaces:** SharedMCTSStore interns planning states, claims node expansion, reserves rollout edges and performs idempotent N/W backups. MCTSRouteNetwork preserves RouteNetwork next_context/incorporate/report/to_dict and exposes drain_cached_events/checkpoint/proposal_count/has_pending.

- [ ] Write deterministic CPU tests for shared N/W and duplicate backup, crashed leases, stock separation, AND/cycle planning, finish boundary, partial checkpoint and no learned value.
- [ ] Run tests: expected import failure before module exists.
- [ ] Implement store and controller, including progressive widening, rollout interleaving, sampling priors and edge-based PUCT.
- [ ] Run tests and complete suite: expected green, with existing skips/warning disclosed.
- [ ] Commit controller/tests; record task result in this plan's ledger.

## Task 2: Runner, persistence and full460 submission

**Files:** Modify scripts/infer_dynamic_electron_trees.py, scripts/run_taiji_frozen_route_search_a100.sh, scripts/analyze_frozen_route_search.py; add MCTS task config and runner tests; update protocol.

**Interfaces:** --search-algorithm=mcts selects MCTSRouteNetwork; --mcts-shared-root binds shared statistics. Existing tree output condition and greedy read-only comparison remain unchanged. Progress includes finished versus initialized percentages and real MCTS backup counts.

- [ ] Write and run failing selector/progress/launcher tests.
- [ ] Wire MCTS, persisted rollouts and cached native-transition journal events into full runner; do not touch single-reaction31k default behavior.
- [ ] Run targeted and complete suites, shell checks and one independent whole-change review. Fix important findings with regression tests.
- [ ] Pin committed runtime into one new full460 ordinary8A100 task, after checking current resources and old task END.
- [ ] Submit with proven donor. Verify actual POD/mount/engines, algorithm manifest, nonempty generations and stdout. Record exact state without calling queued initialization inference success.
- [ ] Commit safe configuration and update memory/status; preserve other dirty files and all prior outputs.
