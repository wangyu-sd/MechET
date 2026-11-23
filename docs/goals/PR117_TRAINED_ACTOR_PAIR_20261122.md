# Held-out comparison of the trained and parent actors

Compare the corrected History-89442 parent and a durable adapter from the running 8-H20 ReFIT-PO task on the same frozen 96 Reference targets. Reuse the parent condition that samples independent routes in the three-method confirmation. Give the trained actor the same independent-route controller, executor, inventory, seed, target order, 64 native calls per target and six inference workers. This costs at most 6,144 additional model calls, about 7.1 A100 hours at the observed development reservation rate.

The update-26 checkpoint is the artifact used to validate admission. It is bound to the original complete record in the running task's checkpoint tree. Its adapter files match their published digests and have the same 372 tensor names, shapes and dtypes as the parent. The PEFT configuration also matches after comparing the target-module names without list order. Seal the selected durable checkpoint before the scheduled evaluation. The preparation retains the exact update-26 proof.

All 96 initial decision inputs were rendered with the same native executor and pinned tokenizer during CPU validation. Within each target pair, the first request ID, prompt token IDs and categorical random stream agree. Later actions and prompts may differ because each actor sees its own executed trajectory. Each output retains its actual actor identity and sampled probabilities.

| Endpoint | Evidence |
| --- | --- |
| First native action accepted | Executor replay accepted the first physical model response. |
| Productive root reaction | At least one root reaction committed within 10, 30 or 64 actual model calls, including the calls spent on rejected or repeated proposals. |
| Full inventory closure | The independent executor discharged every precursor occurrence under the official stock within the common call ceiling. |
| Full inventory closure by call limit | First verified stock closure within 10, 20, 30 or 64 physical model calls. Report paired transitions and intervals on all 96 targets at each limit. |
| Work through first closure | Physical decision calls and generated tokens through the first verified route. Record terminal elapsed time per arm and the observed device alongside them. |

Report paired discordant target counts and intervals across the full 96-target denominator. The comparison estimates the effect of replacing parent weights with the trained adapter under this controller. A genuinely trained control with a matched treatment supplies attribution to a particular objective component if that control becomes available. Runtime is descriptive when the jobs have different batch composition or GPU allocations.

Run `python -m scripts.pr117_actor_transfer --help` for admission or execution and `python -m scripts.pr117_actor_transfer_score --help` for replay. The scorer uses the existing archived native receipt audit and paired estimator on `PYTHONPATH`. Use the shared `reference_trained_actor_update26_prepared_20261122/prepared.json` for the CPU preparation receipt. The existing 96-target confirmation registration remains unchanged.

## Scheduled execution

The November 26–29 comparison uses the original parent bundle
`reference_confirmation_launch_1a100_20261122`. If the protected validation
selection still gives update 64 at launch, pair it with
`reference_trained_actor_launch_1a100_20261123_update64_physical_prefix`.
The two update-64 preparation directories have the same Taiji task flag. The
physical-prefix directory includes the finalizer for the 10-, 20-, 30- and
64-call comparison. Its sealed commands point to its own worker and results.
The November 23 readback checks the parent and trained workers against their
existing admission receipts on the same 96 targets. [Readback and deployment record](https://github.com/wangyu-sd/MechET/pull/117#issuecomment-7084676060)

The shared run directory contains
`reference_phase_b_launch_preflight_20261123/render.py`. It checks the deployment's
manifest, worker command and authorized Taiji template before rendering the
one-A100 configuration. It prints a configuration with credentials omitted
by default, or writes a private shared-storage file with `--output`. The
planned window and actual separate GPU admission govern both submissions.

Invoke `python -m scripts.pr117_actor_transfer_package` with `--parent-launch`, `--trained-checkpoint`, `--trained-checkpoint-sha256` and `--output` to seal a launch bundle. Use the original durable `complete.json` and its published digest. The package copies the parent's verified panel and runner and changes only the actor condition to run on one separately allocated A100. It pins the original parent result location and validates the complete 96-pair scope on CPU before publication.

The Taiji command is recorded in `inputs/deployment.json`. `bash worker.sh --validate` checks the sealed bundle without model generation. The worker uses the same allocation receipt handoff as the parent. Once both runs have completed, `bash worker.sh --finalize` replays their physical receipts and publishes `paired_actor_results.json` atomically. The parent and trained raw outputs remain separately reproducible.

Run `python -m scripts.pr117_admit_allocation <bundle>` from the isolated worktree to verify the sealed one-A100 deployment. During the scheduled window, pass `--instance-id` after the submitted worker publishes `worker_ready.json`. The command checks that exact running Taiji instance, verifies the hostname and physical A100 with an independent device query, and publishes the allocation receipt used by the waiting worker.

## Reading back completed work after an interruption

Preserve the stopped worker's target files, allocation receipt and native
journal. The shared execution directory contains the readback command in
`reference_actor_interruption_readback_20261123/README.md`. Use the actor
and panel registered in the sealed bundle for that worker. The readback
replays published C0 targets and counts issued calls, returned calls and
persisted behavior receipts separately. Charge any issued call to the
recorded physical work, including a call still waiting for a return.

In the archived development replay, the prefix with six published targets
has 180 calls attributed to those targets. Taking the original journal
through the next C0 dispatch records 181 issued and 180 returned, with
the same six targets. Both prefixes reproduce exactly. The complete
original archive replays all 24 C0 targets and 720 calls. Use the
registered paired finalizer once both formal runs have their completed
summaries. The readback is an analysis step for preserved outputs.
