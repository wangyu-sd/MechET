# Resumable Ray MCTS deployment

Status: approved in chat; user explicitly waived further document-review pauses.
Implementation and review/performance fixes pass 908 software tests (11 skipped).
The full3120 CPU-only migration and independent path audit passed. One8A100
continuation was submitted at23:39 CST and is resource-waiting; actual GPU
recovery/generation/throughput verification remains pending. The old31k is END.

## Requested outcome

Deploy the previously discussed Ray Core + existing vLLM acceleration, replace
the mech-USPTO-31k MCTS task with ONE ordinary eight-A100 task, and retain
restartable search evidence. Poll a queued replacement every 600 seconds;
after allocation, verify real generation and measure sustained throughput.
Do not cancel or restart R1/R2 or other training jobs without authorization.

This is an infrastructure change, not a new search method, experiment subset,
actor, training stage, or endpoint definition. No model smoke/pilot evaluation
is requested. Software unit, fault-injection, and integration tests are allowed.

## Evidence and unchanged contracts

The old instance is 8b1d803aa13ad2de01a13d92657a055f, task
meteor_mechet_uspto31k_mcts_rolling_8a100_qy_20261015_01. Its source runtime
is ada3cbe1e3ee044e68e8f0b3b56aed1061cca236. Before stop, the 22:09 CST
snapshot recorded 125/3120 finished targets; exact per-rank rollout and token
counters belong in the stop receipt. All eight ranks
had 390 checkpoint files each, plus manifests and native journals.

Frozen source: data/mech_uspto_31k_full_endpoint_rxnmapper/test.jsonl, SHA256
819215c5d40720afd998aaf77b7b938045d8171a5487b26bf929897f65a957c4.
Retain all 3120 original IDs. This remains the frozen rxn_prod_min-proxy
DIAGNOSTIC protocol; a performance migration must not silently switch products.

Actor: FlowER Stage-II Qwen3-0.6B, adapter_model.safetensors SHA256
40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75,
environment natural_language_electron_event_history_v2. Preserve pinned base
revision, tokenizer/template, executor, sampling, seed, and mean-NLL ranking.
The native policy/executor cache namespace must remain
5df137df20153114e967310ed660270c7c471ccd443ceea1ad3a4d77652e8011.

Keep single-reaction PUCT, existing widening, and ten TOTAL trajectory attempts
including failed/censored attempts. This is not ten successful endpoints and
not recursive stock-connected planning. GT remains post-search scoring only.
No new chemical depth, breadth, import, time, or token stopping cap.

## Architecture

Use Ray Core for long-lived actors and scheduling, not RLlib or Ray Train.
Retain the proven vLLM 0.8.5 / Torch CUDA11.8 stack and Tencent taiji6 donor
image. Pin a compatible Ray release after checking its dependency requirements;
do not upgrade Torch, CUDA, RDKit, or the model to satisfy an unpinned install.

One tree-owner actor per planning namespace owns mutable topology, N/W,
reservations and completed-backup IDs in RAM. Its state-mutating transactions
are serialized. Exact native state, history, inventory, outstanding obligations
and stock identity remain part of the relevant keys. Ray's immutable object
store is not used as an arbitrarily mutable shared dictionary.

Eight persistent GPU owners serve the existing policy. CPU chemistry workers
execute independent requests in parallel; results feed the search without
waiting for a whole generation/execution wave. Batch tree transactions such as
selection plus reservation and path backup; do not create one remote task for
every atom operation. Bound queue capacity to available resources; queue bounds
are scheduling controls, never chemical search-depth limits.

A separate writer persists ordered delta transactions and checkpoints. Ceph
stores durable evidence and recovery records, not hot-path per-node locks.
Local disk/RAM is only a cache. Initial deployment needs no LMDB/Redis service.

## Durability and restart contract

1. Each mutation carries a unique operation ID and monotonically ordered
   sequence. Persist checksum-protected, batched journal records; acknowledge
   durable operations only after the Ceph commit barrier. Independent requests
   may continue while the writer batches operations. Never treat an undurable
   local write as a node-loss-safe committed result.
2. Preserve raw prompts, token IDs, generations, likelihoods, executor outcomes,
   accepted and rejected edges, native states, controller checkpoints, and
   final per-target reports. Record the last committed sequence in snapshots;
   replay later transactions idempotently. Ignore incomplete tail records.
3. Repeated operations and recovery cannot increment N/W, observation counts,
   completed rollouts, or sampled-token counters twice. Conflicting duplicates
   fail loudly. A final report cannot seal before its supporting commits.
4. New output directories do not overwrite old evidence. An explicit resume
   source verifies source/actor/search contracts and shard assignment before
   loading. Runtime implementation SHA is provenance, not permission to reject
   compatible old data or silently reset scientific state.
5. For legacy 31k, restore journals plus controller checkpoints and planning
   statistics, retain finished reports, completed attempts and terminal
   programs, and reconcile pending UID/path reservations with durable backups.
   Resume a pending trajectory from its saved state rather than charging a
   deployment interruption as an additional chemical failure or attempt.
   Also finalize already-goal-complete checkpoints lacking their evaluation
   report without rerunning inference. The stopped full3120 audit found 126
   goal-complete checkpoints but only 125 reports, so this is a real crash
   boundary rather than a hypothetical test requirement.
6. Reconcile crash windows between expansion persistence, native journaling,
   backup and controller checkpointing. Native payloads and durable policy
   receipts are evidence for reconstruction; missing evidence is a migration
   failure to report, not a reason to invent an endpoint or discard an ID.
7. Reject a live legacy writer before ownership transfer. R1/R2 uses a separate
   planning namespace; leave its running instance intact. It may continue to
   share immutable native policy/executor receipts under existing locks.
8. Actor/worker restart and whole-host restart use the same durable recovery
   path. Fail closed on corrupt or mismatched contracts. SIGTERM requests a
   drain/checkpoint, but correctness must also survive an abrupt process kill.

## Verification and rollout

Before submission, software tests must cover batch-versus-serial PUCT behavior,
exact state/history identity, concurrent reservations, duplicate backup,
actor restart, writer failure, incomplete tail, report/checkpoint ordering,
and migration of both finished and pending legacy targets. Test the backup
committed/checkpoint stale and goal-complete/report-missing crash windows
explicitly. Demonstrate node-loss
recovery from the shared durable path, not merely a surviving Python object.

Run the relevant and full repository regression suite. Record the Ray/runtime
versions, source hash, test output and migration audit. No experimental result
is inferred from these software tests. Validate all 3120 target identities and
all eight legacy shard manifests before launching their continuation.

Submit ONE non-elastic eight-A100 task with task_flag and readable name starting
with meteor. Clone only the successful donor's configured private Ceph init;
never print credentials. Keep unbuffered stdout/stderr on the default POD log,
wrapped by the existing heartbeat helper at 45 seconds (maximum 60 seconds).
Log recovery counts, queue occupancy and checkpoint sequences during startup.

Poll the specific replacement instance every 600 seconds while waiting. A
poll timeout does not justify duplicate submission. Once running, verify the
Ceph mount, eight intended GPUs and eight active inference owners, fresh raw
outputs and default-log heartbeats. Measure at least two subsequent progress
intervals, separating resumed counters from newly generated tokens, completed
rollouts and finished targets. Report GPU utilization, CPU occupancy and
persistence backlog. Do not claim speedup from GPU utilization alone.

Completion means: old 31k END; tested Ray deployment actually generating on
eight A100s; compatible recovery verified; queue monitoring performed if needed;
and observed post-start throughput reported. It does not require the full
3120-target evaluation to finish in this deployment task.
