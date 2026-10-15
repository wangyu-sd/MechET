# Committed cached-path recovery, 2026-10-16

The eight-A100 Ray continuation instance `8b1d899ea13fdea001a14037bb080126`
allocated at03:47 CST and failed before new model generation. Ceph, all eight
A100-SXM4-40GB devices and all eight inference owners were observed. Default
POD logs captured45-second heartbeat output and the failure traceback, retained
in `outputs/submissions/uspto31k_mcts_ray_resume_20261015_02/failure_default_log_tail.json`.
No unrelated job was stopped.

The failure was `saved completed program lacks recoverable native context`.
Four old positive backups were durable before their controller/native-journal
updates. The first migration reconciled their counters but omitted the native
terminal; one path also required its cached intermediate suffix. The original
UIDs, paths, proposals and backups were retained, so no model resampling or gold
endpoint is needed to reconstruct that evidence.

The recovery helper now uses the case's historical pending UIDs to find its
committed whole paths. A disposable executor replay must reproduce every exact
recorded context identity, then fills missing native contexts/edges. Existing
edges are reused. No additional backup, attempt, model-token charge or policy
generation is introduced. Previously unmaterialized states on the committed
path are added to observed graph coverage; this is recovered old coverage,
not new sampling or a claim of independent chemical correctness. Ambiguous
terminal likelihood witnesses or missing/corrupt evidence fail closed.

Four executor-backed regression conditions were observed failing before the
repair and passing afterward: current/stale checkpoint crossed with last-step/
whole-cached-suffix journal lag. They verify two consecutive restores, connected
import/FLOW/finish paths, exact terminal Node including NLL and unchanged
attempt/backup/model-cost counters. Full suite:927passed,11skipped,one existing
PEFT warning83.04s. Frozen executor/planner/cache-namespace source files and shell
syntax checks are unchanged/passing.

The independent all3120 native audit completed in150.63s:

| Evidence | Verified |
|---|---:|
| Source IDs | 3,120 |
| Preserved finished reports | 125 |
| Checkpoints to continue/finalize | 2,995 |
| Completed attempts, including finished reports | 8,874 |
| Pending original UID paths | 19,948 |
| Recorded terminal programs, including reports | 4,445 |
| Reconstructed native terminal contexts | 4 |
| Native contexts verified through JSON snapshot/reload | 166,543 |

Affected original IDs: `22230`, `30312` (rank01), `8863` (rank02), `23884`
(rank07), all with prefix `mech-uspto31k-full:test:`. The audit checked the
latest partial-startup evidence plus its compatible ancestors, exact local
attempt/UID-path preservation, unchanged report scoring and native round trips.
It performed no model generation, acquired no active rollout leases and did not
write to source evidence. It used a separate read-through store for the four
old WAL backups; this is software recovery verification, not a new benchmark.

Audit record:
`outputs/analysis/mcts_native_resume_audit_20261016_02/audit_complete.json`.
The date-specific auditor is `scripts/audit_mcts_native_population_20261016.py`;
its existing-output guard preserves the original audit record.

The replacement must read the partial
`outputs/eval/mech_uspto31k_full_mcts_ray_20261015_02`, fall back through the
completed migration to the original rolling output, and reopen the SAME
persistent Ray WAL. All3120 frozen min-proxy DIAGNOSTIC IDs, FlowER Stage-II
Qwen3-0.6B actor, native cache, PUCT and ten TOTAL attempts remain unchanged.
Actual new generation and two throughput windows still require deployment.
