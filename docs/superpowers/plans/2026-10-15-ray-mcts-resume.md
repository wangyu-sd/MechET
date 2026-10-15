# Resumable Ray MCTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. User explicitly waived further document review gates.

**Goal:** Continue the stopped full3120 MCTS from its saved state on one eight-A100 instance, accelerating tree access with Ray and verifying restart and throughput.

**Architecture:** Preserve existing planner source hashes and scientific contracts. Add a compatible resumable controller, a Ray-owned in-memory tree with ordered durable transactions, and integrate them into the rolling runner. Old artifacts stay read-only.

**Tech Stack:** Python3.11, existing RDKit, Torch2.6 CUDA11.8, vLLM0.8.5, compatible pinned Ray Core.

**Spec:** docs/superpowers/specs/2026-10-15-ray-mcts-resumable-deployment-design.md

## Global Constraints

- All3120 frozen source IDs; same actor and native-cache SHA contracts.
- Ten total attempts; no new chemical search cap; no gold feedback.
- No unrequested model smoke or unrelated task cancellation.
- Native journals, raw generations, likelihoods, paths and exactly-once backups persist.
- One ordinary8A100 Tencent successful-donor task, meteor prefix,45-second default stdout heartbeat.
- User approved the in-chat design and explicitly said not to stop for file review.

## Review Focus

- A backup committed before the controller checkpoint must not count twice after restore.
- A completed checkpoint without its report must finalize without new sampling.
- A pending selected expansion must retain its UID/path, not become a censored attempt.
- Journal/graph lag must recover native contexts without changing actor or endpoints.
- An actor/host crash must recover durable transactions and refuse a second live writer.

### Task 1: Compatible controller and full-population recovery

**Files:** Create scripts/mcts_resume.py and tests/test_mcts_resume.py; modify scripts/infer_single_reaction_mcts.py.

**Interfaces:** restore_single_controller(journal, checkpoint_path, mcts_store, shared_store) returns a resumable SingleReactionMCTS-compatible controller. Runner accepts explicit --resume-from and validates original manifests.

- [x] Write failing executor-backed tests for pending paths, stale checkpoint/committed backup, goal-complete finalization and contract mismatch. Run targeted pytest; Expected: missing implementation fails.
- [x] Implement restoration with unchanged original planner/hash files, reclaim saved UID leases, hydrate contexts, reconcile completed backups, retain completed-program sets.
- [x] Test targets do not lose history or reset counts; Expected: targeted tests pass.
- [x] Integrate runner recovery without overwriting original artifacts, carry prior reports/cost counters with separate resumed/new work accounting.
- [x] Run whole pytest suite; Expected: no new failures. Commit task and record result.

### Task 2: Durable in-memory tree and Ray ownership

**Files:** Create scripts/ray_mcts_store.py, tests/test_ray_mcts_store.py and dependency lock/launcher changes.

**Interfaces:** Ray store provides existing intern/snapshot/expansion/reservation/backup API; preserves contract, supports reopen and immutable legacy read-through. Recovery calls use resume_rollout(uid) and backup_record(uid).

- [x] Write failing tests for old/new topology equivalence, reserve/backup idempotency, transaction replay, incomplete WAL tail, corrupt records, ownership and real Ray restart.
- [x] Implement memory store with serialized state mutation, batched durable journal writer and snapshot/replay; Ray actors separate tree ownership from GPU workers.
- [x] Test source contracts remain byte-compatible and old tree files unchanged; Expected: targeted pass.
- [x] Verify exact pinned Ray runtime in a local isolated dependency overlay, no Torch/CUDA upgrades.
- [x] Run whole pytest suite; Expected: no new failures. Commit task.

### Task 3: Full continuation deployment and monitoring

**Files:** Update pinned task launcher; add one meteor8A100 config, submission/monitor receipts and status records.

**Interfaces:** Explicit old-output resume source; new output; immutable committed runtime. Queue watcher polls the exact instance every600seconds and switches to startup/throughput validation.

- [x] Test launcher resume arguments, output guards, signals and no secret output. Expected: software tests pass.
- [x] Audit all3120 saved IDs/contracts and recover them without model generation. Expected: no lost IDs or reset attempts.
- [x] Run final whole-branch review and full suite. Expected: no Critical/Important unresolved.
- [x] Submit exactly one ordinary8A100 from successful Tencent/Ceph donor; verify actual instance state.
- [ ] Monitor every600seconds if queued, then verify eight real engines/default logs/new raw results and measure two progress windows.
- [ ] Record observed throughput, recovery provenance and remaining limitations. Completion requires real deployment, not only create/start success.

### Task3 recovery-start interruption boundary (implementation supplement)

The requested abrupt-host-restart contract includes interruption while restoring
the old population, not only interruption after inference begins. This is an
implementation correction, not a new experiment, actor or search condition.

- [x] Reproduce missing current report/checkpoint and an empty newly-created
  journal with executor-backed software tests; both failed as expected.
- [x] Restore the newest saved checkpoint with the nearest initialized compatible
  ancestor journal; reuse ancestor reports and never write to source evidence.
- [x] Publish a checksum-protected origin companion outside the fresh output
  directory, so even missing rank manifests can be recovered without weakening
  the existing-output guard. Validate output binding and scientific contracts.
- [x] Test actual Ray bootstrap from such a partial startup, including lookup of
  the original checkpoint and reopening the same durable store.
- [x] Run the complete suite and audit artifact selection for all3120 original
  IDs. Add the current queued job's compatible origin metadata without changing
  its submitted runtime, source data or queue position; no new GPU job is needed.

### Recovery progress metadata follow-up

- [x] Treat missing startup progress as optional accounting, not missing search
  state; fall back to the nearest compatible recorded ancestor.
- [x] Normalize carried historical token/forward/time counters once, without
  summing copied snapshots from every ancestor. Regression reproduced lost
  prior costs; targeted origin/runner suite23passed and all8 real shard metadata
  comparisons passed without model generation or source writes.
- [x] Run the final whole suite and commit this accounting-only follow-up.

Final accounting verification:923passed/11skipped/1existing PEFTwarning101.98s;
shell syntax and frozen planner/executor comparison passed. Runtime deployment
and two observed throughput windows remain unverified; Task3 is not complete.

### Task3 actual-deployment cached-path failure

- [x] Read the actual default POD traceback after the first instance failed
  during restoration, before model generation; preserve its partial output.
- [x] Reproduce positive backup/native-journal lag and a wholly cached suffix
  in executor-backed tests; cover two consecutive restores and exact path/NLL.
- [x] Reconstruct only committed historical UID paths, not new model samples;
  retain existing edges and original attempt/backup/model-cost accounting.
- [x] Verify all3120 latest/ancestor records and166543 native JSON round trips;
  all4 missing terminals recovered,125reports/2995CPs/8874attempts/19948pending
  paths/4445terminal programs preserved,150.63s,no model generation/source write.
- [x] Final full suite927passed/11skipped/1existingwarning83.04s; frozen source
  contract comparison and shell syntax passed.
- [x] Commit repair and submit one fresh-output8A100 continuation of the partial
  failed instance (0ec4ff8/config586a977,instance8b1d8242a13fde4101a1413c9f36031d).
- [ ] Verify actual post-repair eight-worker recovery, fresh default logs/new
  model generations and two throughput windows; initialRUNNING alone is not done.
