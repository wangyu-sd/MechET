# Corrected 0.8B fallback and actual results

The user authorized using 0.8B first while the 8B environment is blocked. This is a separate Qwen3.5-0.8B condition; it does not produce corrected Qwen3-8B weights.

## Verified results before this submission

- 8B retry03 ended unsuccessfully after all eight training ranks exited at 2026-11-14 15:14:19 CST: `PackageNotFoundError: No package metadata was found for bitsandbytes`. No new checkpoint or optimizer update is confirmed. GPU fit and training throughput were never measured, so inability to train 8B on eight H20s is not established.
- The previous corrected 0.8B v3 run completed 21 optimizer updates of 371 scheduled rounds before the user stopped it on November 13. Its final completed round covered 64 targets and 256 independent route episodes: 5,084 model decisions, 265 committed reactions, 19 inventory-closed episodes, zero reference-supported closures, six informative reward groups and 13 censored groups. Collection wall time was 3,351.99 seconds. These are training-round observations, not held-out improvement or full-population success rates.
- The earlier explicitly authorized v8 diagnostic used two fixed TRAIN targets and two replicas per target. Committed reactions increased from 3 to 8, and inventory-closed episodes from 0 to 2, both on the same target. Reference-supported closures remained zero. The policy still proposed 21 repeated rejected actions, although their duplicate executor calls were skipped. No convergence or GPU speedup is established.

## Formal fallback condition

Reuse the previously exercised fixed 0.8B runtime instead of the 8B image-only dependency path. Start from corrected FlowER Stage-II History-89442, adapter SHA `2c98a9bff3afeaffbf2ee2733885cccc68e936f9310fd9f46ba4ee2211e5f453`. Retain old v3 round21 and all 8B failure evidence; do not relabel those artifacts as this condition.

The new condition uses 4 H20, full external TRAIN 23,730 targets and fixed VALID 2,986 targets, G4, seed17, one epoch and the existing Dr.GRPO objective and hidden reference reward. R1/R2 and SimpRetro remain protected evaluation data. The initial actor baseline remains deferred until after training; scheduled full VALID checks retain the original protocol.

Set the explicit observation contract to `route_action_feedback_v8`. It preserves complete raw history, scopes historical atom aliases, supplies current source bonds and constraints, resets each newly selected reaction target, advertises FINISH only when eligible, and reuses the receipt for an equivalent rejected action in the same state. Model proposals and probabilities remain recorded; this does not claim to prevent repeated policy proposals.

Source is the existing immutable c43bba1 snapshot. No unrelated dirty MCTS edits, new model packages, new reward, critic or teacher mechanism labels are included. All data, caches and runtime artifacts remain on Ceph. The failed 8B job is END; the fallback requests only four GPUs under the authorized eight-H20 ceiling.

## Gates and follow-on work

Before submission: verify the complete actor/data/source/runtime bindings, retain a separate non-submittable public template and a private 0600 mount configuration, and verify the deployed contract. The 34 relevant action-feedback, actual training factory, memory-integration and curation tests passed in 94.39 seconds. These are CPU software checks, not model results.

After submission: distinguish scheduler state, actor-ready, collected reactions and actual saved parameter updates. Record collection/update timings and useful reward groups. Compare held-out results only after the prescribed full VALID evaluations exist.

This first fallback run exercises v8 multistep learning and collects actual TRAIN recovery evidence. Verified recovery curation, memory-conditioned SFT, G1–G6 framework work and RSI iteration remain separate gates; they are not automatically complete when this job starts.

The current task receipt is published separately in `configs/ops/qwen35_08b_fallback_20261114.json` after actual platform submission.


## Submission observation

Task `meteor_mechet_stage2_external_route_drgrpo_4h20_zjk_20261114_action_v8_02`, instance `8b1d8208a1cfaa4801a1d8ff12bb110c`, was created at15:39:18CST and reached platform TRAINING_RUNNING at15:39:47. This scheduler state does not confirm model loading, sampling or a parameter update. Contract SHA `07f202555705fcba65ff2b60b8e37a0c9c85aa24fcb5c299340e5fba7ab21262`; independent static review confirmed all390 deployment sources equal the immutable source and found no Critical/Important blocker. The health monitor reads actual route checkpoint receipts, and its schema handling was checked against retained real round21, wrong-contract rejection and an absent new checkpoint.
