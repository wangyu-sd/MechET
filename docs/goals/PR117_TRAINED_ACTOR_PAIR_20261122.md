# Held-out comparison of the trained and parent actors

Compare the corrected History-89442 parent and a durable adapter from the running 8-H20 ReFIT-PO task on the same frozen 96 Reference targets. Reuse the parent condition that samples independent routes in the three-method confirmation. Give the trained actor the same independent-route controller, executor, inventory, seed, target order, 64 native calls per target and six inference workers. This costs at most 6,144 additional model calls, about 7.1 A100 hours at the observed development reservation rate.

The update-26 checkpoint is the artifact used to validate admission. It is bound to the original complete record in the running task's checkpoint tree. Its adapter files match their published digests and have the same 372 tensor names, shapes and dtypes as the parent. The PEFT configuration also matches after comparing the target-module names without list order. Seal the selected durable checkpoint before the scheduled evaluation. The preparation retains the exact update-26 proof.

All 96 initial decision inputs were rendered with the same native executor and pinned tokenizer during CPU validation. Within each target pair, the first request ID, prompt token IDs and categorical random stream agree. Later actions and prompts may differ because each actor sees its own executed trajectory. Each output retains its actual actor identity and sampled probabilities.

| Endpoint | Evidence |
| --- | --- |
| First native action accepted | Executor replay accepted the first physical model response. |
| Productive root reaction | At least one root reaction committed within 10, 30 or 64 actual model calls, including the calls spent on rejected or repeated proposals. |
| Full inventory closure | The independent executor discharged every precursor occurrence under the official stock within the common call ceiling. |
| Work through first closure | Physical decision calls and generated tokens through the first verified route. Record terminal elapsed time per arm and the observed device alongside them. |

Report paired discordant target counts and intervals across the full 96-target denominator. The comparison estimates the effect of replacing parent weights with the trained adapter under this controller. A genuinely trained control with a matched treatment supplies attribution to a particular objective component if that control becomes available. Runtime is descriptive when the jobs have different batch composition or GPU allocations.

Run `python -m scripts.pr117_actor_transfer --help` for admission or execution and `python -m scripts.pr117_actor_transfer_score --help` for replay. The scorer uses the existing archived native receipt audit and paired estimator on `PYTHONPATH`. Use the shared `reference_trained_actor_update26_prepared_20261122/prepared.json` for the CPU preparation receipt. The existing 96-target confirmation registration remains unchanged.

## Scheduled execution

Invoke `python -m scripts.pr117_actor_transfer_package` with `--parent-launch`, `--trained-checkpoint`, `--trained-checkpoint-sha256` and `--output` to seal a launch bundle. Use the original durable `complete.json` and its published digest. The package copies the parent's verified panel and runner and changes only the actor condition to run on one separately allocated A100. It pins the original parent result location and validates the complete 96-pair scope on CPU before publication.

The Taiji command is recorded in `inputs/deployment.json`. `bash worker.sh --validate` checks the sealed bundle without model generation. The worker uses the same allocation receipt handoff as the parent. Once both runs have completed, `bash worker.sh --finalize` replays their physical receipts and publishes `paired_actor_results.json` atomically. The parent and trained raw outputs remain separately reproducible.

Run `python -m scripts.pr117_admit_allocation <bundle>` from the isolated worktree to verify the sealed one-A100 deployment. During the scheduled window, pass `--instance-id` after the submitted worker publishes `worker_ready.json`. The command checks that exact running Taiji instance, verifies the hostname and physical A100 with an independent device query, and publishes the allocation receipt used by the waiting worker.
