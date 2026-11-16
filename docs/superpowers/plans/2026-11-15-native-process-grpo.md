# Native process-GRPO implementation plan

> Implement inline using executing-plans; the user has authorized design through full job submission. Keep a durable progress ledger. Existing isolated worktree is MechET-earho-publication-20261026; unrelated dirty MCTS files must not enter the deployment.

**Goal:** Restore the corrected SFT interface and train verifiable fine-grained process-GRPO on the complete external multistep dataset.
**Architecture:** Versioned native observation extension, opt-in post-rollout process scorer, per-action group return-to-go advantages and unchanged token clipping/KL kernel. Legacy behavior remains available under its existing contract.
**Spec:** ../specs/2026-11-15-native-process-grpo-design.md

## Constraints and review focus

- Same History89442 actor/reference, 23730TRAIN/2986VALID, G4,seed17,oneepoch; four-H20 new condition, at most eight total. No smoke/pilot.
- Ceph for every artifact/cache/test tempfile. No hidden reference actions in prompts; unknown resource outcomes never become failure labels.
- Watch unsupported ancestors, repeated AND occurrences, rejected-action graph mutation, stale feedback and forged restore, variable-length advantages and distributed normalization.

## Tasks

- [ ] 1. Write failing reward-ledger and causal-group-advantage tests. Add opt-in process scoring with per-commit trusted progress, bounded monotone validator costs and exact reward conservation. Keep legacy scalar scorer output identical by default.
- [ ] 2. Write native-observation/failure/restore tests. Implement route_sft_aligned_feedback_v9 using native SYSTEM/USER/tools and archived full ledger. Audit actual full corrected VALID schema/prefix bindings.
- [ ] 3. Test actual training factory, dynamic oracle selection and per-record advantage integration; retain scalar legacy path and independent double-precision tensor gradients.
- [ ] 4. Add strict named protocol validation and documentation/formulas/source references. Run focused regression suites, full saved-evidence audit and an independent final source/config review.
- [ ] 5. Freeze only named source changes on the known v8 runtime. Verify full actor/data/runtime/manifests, preserve old v8 checkpoint, create/start full four-H20 job with genuine private mount config. Record actual platform/actor/collection state and update PR.

## Ledger

- 2026-11-15: Research primary DeepSeekMath, Dr.GRPO, GiGPO, PAV and potential-shaping papers completed. SFT source confirms history_contains_failed_actions=false, two native input messages and compact accepted actions. Optional user preference about failure-only short tool receipt is pending; default proposal is an explicit extension with unchanged native SYSTEM/final USER.

- Reward TDD:8RED missing module ->22affectedPASS41.69s; review prompted retention of old0.2 partial ceiling and eta=.05/L so total validation cost stays below one trusted edge. New scale regressionsRED before repair; final green run pending. No GPU result.

- Native observation TDD:4RED/1legacy-pass ->33affectedPASS7.49s. Exact native SYSTEM/current USER/static tools on accepted prefixes, failure-only genuine short tool receipt and full replay restore implemented. All original events remain archived.
- Reward scale final:keep originalq=.2*S/L; eta=.05/max(L,1), so max total error cost is one quarter of one trusted-edge reward.22affectedPASS30.91s. Process optimizer integration2RED observed; green run pending.

- Full actual SFT VALID audit PASS21355decisions/2751reactions, native system/user/tools plus actual Qwen3.5 token-prefix IDs exact; max reference-prefix length2491. Runtime Python3.11/pinned tokenizer/RDKit; no model forward.
- Full saved v8 audit PASS18completedrounds/72rankfiles/4608episodes/89674records: base reference scorer exactly reconstructed; group censor unchanged614; old nonconstant65→new reward-nonconstant467 and step-coefficient-nonzero470. This is an offline signal check, not on-policy improvement, and old receipts are never trained with new observations.
- Independent review found unstarted snapshot restore and failure chat delimiter re-injection; both reproduced RED (incl actual tokenizer6vs5role markers), repairs implemented, final combined check pending.
