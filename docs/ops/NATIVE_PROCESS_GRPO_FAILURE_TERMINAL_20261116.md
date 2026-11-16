# MechET Native Process GRPO: First-Rejection Termination

**Date:** 2026-11-16  
**Scope:** Follow-up contract for the next training run. This change does not alter, stop, or restart the currently running training task.

## Why this change is necessary

The PR #107 episode implementation permits up to eight consecutive recoveries after an Executor rejection. In the reported 28-round audit, 3,096 incomplete routes ended after repeated FINISH attempts, and 4,694 incomplete routes contained repeated same-state, same-semantic failures.

This is not evidence of an off-policy failure replay buffer. The frozen contract explicitly excludes `history_file`, `train_file`, and `teacher_actions`; the repeated attempts are new on-policy generations within the same Episode. The environment returns the latest rejected action and Executor error as a new observation, then asks the policy to generate again.

That recovery prompt is outside the clean SFT-native observation distribution. Repeatedly asking a 0.8B policy to repair the same rejected action is not a useful chemistry-learning objective.

## New process-GRPO contract

For `route_sft_aligned_feedback_v9`, the next run must set:

- `max_consecutive_recoveries = 1`
- first non-resource Executor rejection terminates the Episode;
- the rejected action remains in the behavior-token ledger and receives its normal process penalty;
- no subsequent model generation is requested for that failed Episode;
- resource/context/GPU censoring remains censored and is not treated as chemical failure.

The process protocol validator now rejects a native process-GRPO contract with any recovery ceiling other than one. The contract generator also emits the process algorithm, SFT-native observation contract, process reward configuration, action-return-to-go estimator, and fail-fast recovery ceiling together. This removes the previous mismatch in which the source snapshot's generator defaulted to legacy route-history/Dr-GRPO settings while the submitted formal contract described native process GRPO.

## What is deliberately unchanged

- No reward weights or reward formula are changed.
- IMPORT and FLOW do not receive a positive reward merely because the Executor accepted them.
- Reaction-boundary progress, route-terminal reward, and bounded validator cost remain unchanged.
- No teacher actions, MCTS winner replay, or historical failure examples are introduced.
- Fixed VALID targets, candidate count, decoding settings, and validation schedule remain unchanged.

## Deployment and safety

Apply `patches/route_sft_process_grpo_failure_terminal_20261116.patch` only after the original PR #107 runtime patch has been applied to the corresponding frozen runtime source tree. Re-freeze the source inventory and contract hashes, then create a new formal submission manifest.

The currently running task is pinned to its existing source/contract snapshot. This follow-up branch must only be used for a subsequent run. Do not edit its live contract, checkpoint, or output directory.

## Required checks before the next submission

1. `verify_contract` accepts `max_consecutive_recoveries=1` and rejects 8.
2. A rejected IMPORT, FLOW, FINISH, or malformed action produces exactly one recorded attempt and no second model request.
3. Resource-censored Episodes still return `reward=None` and censor their complete G4 group.
4. Accepted FLOW followed by FINISH still proceeds normally and starts a fresh native observation for the next precursor occurrence.
5. The new formal manifest records the actual runtime commit, source archive hash, contract hash, and `max_consecutive_recoveries=1`.

This patch is a rollout-control correction, not evidence that the current policy has learned better chemistry. Reward decomposition and fixed VALID results must still be measured from the real training artifacts.
