# Ray/vLLM continuation of the stopped full3120 MCTS

## Deployment state

At 2026-10-15 23:39 CST, submitted exactly one ordinary eight-A100 task in
`AI4LSLab_KEMO_QY_A100`, location `qy`:

- Task: `meteor_mechet_uspto31k_mcts_ray_resume_8a100_qy_20261015_02`.
- Instance: `8b1d899ea13fdea001a14037bb080126`.
- Pinned runtime: `7bbc16f7bb314c5a059ac5e08e9b1c398d70512d`.
- Image: `mirrors.tencent.com/whaleywang/metabo:taiji6`.
- Proven successful-donor private Ceph init; no credentials enter Git.
- Unbuffered default POD stdout/stderr and a 45-second heartbeat.

Initial authoritative `instance_list` readback is `TRAINING_RESOURCE_WAITING`.
Submission is not successful inference. A read-only monitor checks the exact
instance every 600 seconds. Actual allocation, recovery, GPU generation and two
throughput windows remain unverified at this record's creation.

The first 600-second readback at23:50 CST remains resource-waiting. The exact
scheduler message is `WAITING_8B1D81A3A13FDE3301A1402CFA5E010C`: this instance
is behind the earlier eight-A100 `crn_pr25_sequence_head_8a100_20261015_01`.
That predecessor reports `底层资源不足，请稍侯重试或排队等待。` Task readback
confirms the intended application group and `location=qy`; no incorrect-region
evidence was found. Group quota112/using87/waiting16 and cluster-level nine
available A100 hosts do not prove a group-eligible full eight-GPU allocation.
No unrelated job was stopped or modified to obtain resources.

## Preserved scientific contract

This changes tree storage/coordination and adds compatible checkpoint recovery;
it does not change the actor, source products, search algorithm or sampling.
All 3,120 original test IDs remain. The source is still the frozen
`rxn_prod_min`-proxy **diagnostic**, not an audited principal-product benchmark.
The actor is FlowER Stage-II Qwen3-0.6B, adapter SHA
`40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75`.
Ten total single-reaction trajectory attempts include invalid/censored attempts;
they are not ten successful endpoints and not recursive stock-connected routes.
GT remains post-search scoring only. The native cache namespace, planner source
hashes, tokenizer/history, seeds and mean-NLL ranking are unchanged.

## Completed all-population recovery

The original stopped task remains END. Its source is preserved read-only:
`outputs/eval/mech_uspto31k_full_mcts_rolling_20261015_01`.
No unrelated task was stopped; fresh readback left R1/R2 and joint-import-flow
training at `TRAINING_RUNNING`.

Full CPU-only migration completed in 1,478.37 seconds, without model generation:

| Quantity | Original stopped checkpoints | Recovered |
|---|---:|---:|
| Unique source IDs | 3,120 | 3,120 |
| Reused finished reports | 125 | 125 |
| Controllers to continue/finalize | 2,995 | 2,995 |
| Completed attempts | 8,870 | 8,874 |
| Pending rollout paths | 19,952 | 19,948 |
| Collected terminal contexts | 4,441 | 4,445 |

Four shared backups were already durable before their original controller
checkpoints advanced. Recovery reconciled those four exactly once; it did not
generate four new trials. Completed-plus-pending totals are unchanged. Formal
terminal contexts are not claims of chemically correct or GT-matching endpoints.

An independent read-only audit verified all 3,120 IDs, exact finished reports,
and byte-equivalent UID/sid/path records for all 19,948 remaining rollouts.
The four removed pending UIDs each have an original durable backup. The original
completed-but-unreported case remains eligible for no-sampling finalization.

Migration tree metrics: 141,004 planning nodes read, WAL sequence 4 committed 4,
writer queue 0, eight registered recovery workers. The reopened Ray store uses
the same append-only WAL plus immutable original-node read-through; it does not
interpret the WAL directory as legacy node JSON.

## Software verification

Final full suite: **908 passed, 11 skipped, one existing PEFT warning** in 94.50s.
This includes real Ray actor restart and a separate-process bootstrap client,
abrupt process loss, torn WAL tails, writer failure, exact-once backup, UID
admission before cached execution, full preserved native proposal batches,
legacy live-owner fencing/dead-worker transfer and unrelated-node concurrency
during a slow legacy read or lock probe. Shell syntax and `git diff --check`
also passed. These are software checks, not model smoke evaluations.

Independent review found three Important recovery gaps; all were reproduced
with failing tests and repaired. Full CPU migration attempts 01/02 exposed
global-lock Ceph serialization and were intentionally stopped after diagnosis;
their partial outputs/failure receipts are retained. They are not model results.
The successful third migration moves cold reads and per-UID file probes outside
the global tree lock. A same-image running POD reports a soft/hard file limit
of 1,000,000, sufficient for the central legacy lease fences.

## Durable evidence outside Git

All paths below are relative to `/aaa/fionafyang/buddy1/whaleywang/MechET/`:

- `outputs/eval/mech_uspto31k_mcts_ray_migration_20261015_03/migration_complete.json`
- `outputs/analysis/mcts_ray_resume_path_audit_20261015_03/audit_complete.json`
- `outputs/submissions/uspto31k_mcts_ray_resume_20261015_02/submission.json`
- `outputs/submissions/uspto31k_mcts_ray_resume_20261015_02/queue_checks.jsonl`
- New inference output: `outputs/eval/mech_uspto31k_full_mcts_ray_20261015_02`

Raw model receipts, old native journals/trees, source data and actor weights are
not copied into Git. No GitHub push, PR or merge was requested by this deployment.

## 2026-10-16: interruption during recovery itself

Fault injection reproduced two additional startup boundaries: a partially
restored output may not yet contain its inherited finished report/checkpoint,
or may have a newer checkpoint but an as-yet uninitialized journal. The new
entry follows the compatible resume-source chain, retains the newest checkpoint
and chooses the nearest initialized journal without creating source headers or
locks. Missing/corrupt committed records and incompatible contracts still fail
closed; this is not permission to regenerate a target from its product.

A checksum-protected `.resume_origin.json` sibling binds output, original
source, eight shard manifests and persistent store identity. It can recover an
interrupted startup even before any worker manifest was published, without
creating the fresh output directory or weakening its overwrite guard. Origin
publication is fenced and refuses conflicting metadata. A real Ray server plus
separate client passed the no-worker-manifest startup test.

The updated full software suite passed **920 tests, 11 skipped, one existing
PEFT warning**, in82.45s. The independent full3120 artifact-selection audit
passed in21.32s: exactly125 original reports /2995 original checkpoints,
8874 completed attempts /19948 pending paths /4445 terminal contexts. No model
generation occurred and the frozen executor/planner files are unchanged.
Audit: `outputs/analysis/mcts_resume_lineage_audit_20261016_01/audit_complete.json`.

The queued inference instance remains pinned to7bbc16f; it has not been silently
retargeted to edited working-tree code or resubmitted. The new entry is for a
subsequent explicit continuation if necessary. Its compatible origin companion
was published at01:26 CST without creating the inference output directory or
changing the queue position. Publisher2ff5943 and inference runtime7bbc16f
remain separately recorded in
`outputs/submissions/uspto31k_mcts_ray_resume_20261015_02/resume_origin_publication.json`.
Companion SHA256:
`b66e6eb2fc80f9db4202a4f42133d82bd29ec0ca4a2c594ba678b9772e87bf07`.

A final accounting boundary was reproduced and fixed: missing startup progress
does not invalidate saved search state, and repeated resumptions carry recorded
prior token/forward/time costs exactly once. The reader takes the nearest
compatible snapshot rather than summing copied ancestor snapshots; unavailable
costs are explicitly marked unavailable. All8 actual shard metadata comparisons
passed, with no model generation or source writes. Targeted origin/runner tests:
23passed. Final whole suite: **923passed,11skipped,one existing PEFT warning**,
101.98s. Shell syntax, frozen-source comparison and `git diff --check` passed.
These later recovery-entry fixes do not change the queued immutable runtime.

## Remaining runtime gates

- Verify actual POD, Ceph, eight intended GPUs and eight inference owners.
- Verify all eight source/actor/backend manifests and recovered counters.
- Verify fresh default-log output and newly saved non-cache model generations.
- Measure at least two subsequent windows of new tokens, completed attempts,
  finished cases, CPU/GPU occupancy and persistence backlog; do not infer a
  speedup from restored historical counters or a GPU-utilization snapshot.

Explicit resume after interruption is supported. The public config requests
fault tolerance off, but task-detail readback reports it on; automatic recovery
is NOT verified. The existing-output guard refuses silent reuse/overwrite, so
an interruption still requires an explicit fresh-output continuation. No claim
is made to attribute original pre-stop rollout UIDs
that were never saved to any controller, or to recover unrecorded GPU work.
