# Shared-root liveness probe overhead

This is a software performance diagnostic, not a model experiment or cluster
speedup claim. No source tree, model, sampling or scientific contract changes.

The saved root `fe53b348...e27ddf7f` has 1,893 reservations and 2,613 completed
backups. The deployed0ec4ff8 snapshot path checks process creation time and
liveness separately for every reserved UID. Multiple UIDs have the same owner.

The follow-up memoizes these real checks only within one snapshot/batch, keyed
by owner, PID and process creation time. Independent reads and direct ownership
claims still recheck liveness. Legacy cross-process locks remain per UID, and
the original per-UID lock order is retained. A process dying during a batch may
remain observed-alive until that read ends, just as a process can die after an
individual liveness check; a claim never uses this read-local memo.

Five local reads of the real saved root, assigning its original reservations
to eight process-owner identities, produced byte-equivalent snapshot content:

| implementation | median seconds/read | process probes/read |
|---|---:|---:|
| deployed0ec4ff8 | 0.130291 | 5,679 |
| read-local owner memo | 0.011703 | 24 |

The snapshot operation is11.13x faster in this local microbenchmark. This is
not an11.13x end-to-end inference claim. Evidence:
`outputs/analysis/mcts_ray_owner_probe_20261016/software_snapshot_benchmark.json`.
The original saved root is read-only; no model generation occurs in this test.

Tests first reproduced120/240 process probes for40 reservations/two owners in
single/batched reads. The repair passes both performance regressions and the
PID-incarnation/next-read freshness guard. Targeted39passed; whole suite
930passed/11skipped/one existing PEFT warning,83.56s. An initial observer fixture
incorrectly replaced the psutil class with a function; this was corrected to
wrap its real constructor before observing the intended failing probe counts.

Deployment status: **not active in the currently running_03 task**. That task
remains pinned to0ec4ff8 and is already generating from the saved tree. Do not
silently hot-swap its runtime or claim these measurements as its throughput.
Finish its requested recovery/throughput verification before deciding whether
a subsequent explicit continuation merits another interruption. This follow-up
does not authorize stopping any unrelated job or starting a second GPU task.
