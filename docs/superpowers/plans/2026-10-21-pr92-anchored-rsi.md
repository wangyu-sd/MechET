# PR92 Real-data Anchored RSI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline, with test-driven-development.

**Goal:** Execute PR92 S0–S3 on actual preserved FlowER/Qwen data, not a design-only update.
**Architecture:** Leave PR91 v0 untouched. Add task-aware net productivity, conservative common-state preferences and TRAIN-only replay; extend the existing selected-action scorer for capped DPO+replay control updates. Reuse rolling vLLM/CPU tools under fixed v1 seeds, independent greedy/stochastic validation, and per-target work accounting.
**Tech Stack:** RDKit2026.03.4, pinned Qwen3-0.6B, vLLM0.8.5/CUDA11.8, PEFT/PyTorch.
**Spec:** docs/goals/MECHET_RSI_REAL_ANCHORED_CORRECTION_CODEX_GOAL_20261021.md

## Global Constraints

- Stacked PR92 branch only, no merge, no PR91 task cancellation/hot-swap, no paper/Graph/CRN changes.
- Same TRAIN128/VALID32 IDs; separate output `outputs/agent/mechet_rsi_real_anchored_smoke_20261021`; no TEST.
- Original R0 SHA40204...7e9de75 and provisional F0 checkpoint54000/38e663...aa89e frozen. No v0 R1/R2 parent substitution.
- Strong chosen needs executed terminal, independent F0 hit and private task-compatible net progress; mismatch alone/unchecked/censored stays unknown.
- Exact Stage-II current-state+compact-history prompt and complete EOS tool-call replay. Private reference classes/answers never enter actor prompts.
- Source reaction cap2pairs, max4exposures/pair, <=20steps derived from actual data, beta.1/lr2e-6/replaylambda.25. Separate per-loss gradients/costs.
- M0/R1v0/M1replay/M2DPO/M3mixed, greedyK1 and stochasticK2, shared model-independent decoding seeds. Fresh M3 D1 and matched staticD0 continuation only when informative.

## Review Focus

- Different private atom-map namespaces for recorded references versus generated states: classify each internally, never compare addresses across namespaces.
- Acid/base-only positive loopholes without rejecting genuine redox, stereo or necessary proton-transfer intermediate events.
- Replay source-ID, exact history prompt, full assistant EOS and actual replay execution; never rebuild a gold action at a guessed state.
- Common-state pairing/duplicate successors, reaction caps, tiny pool exposure limits, and branch-specific gradient retention.
- New cache/seed/decoder identities, incomplete cost records and complete per-target denominators under token censors.

### Task 1: Productivity, preferences and real offline audit
**Files:** `src/mechet/rsi_productivity.py`, `src/mechet/rsi_preference_v1.py`, `scripts/audit_rsi_v1.py`; tests `test_rsi_productivity.py`, `test_rsi_preference_v1.py`.
**Interfaces:** `net_reaction(target, precursor)` returns class/evidence/unknown in its own map namespace; `productivity(episode, reference_class)` returns independent progress label; `strong_pairs(episodes,max_per_reaction=2)` uses exact prompt IDs and strong-vs-unknown labels; `paired_candidates(episodes,limit)` allocates per product/common-state.
- [ ] RED tests: two actual shortcut fixtures, allowed acid/base/redox/stereo and intermediate proton transfer; mismatch/unchecked never hard-negative; same-state/correlated pair caps and complementary allocation.
- [ ] Implement minimal classifiers/pairs and immutable offline audit from actual v0 D0/D1, frozen proof TRAIN reference crosswalk.
- [ ] GREEN targeted tests and real S0 source hashes/counts/old-pair reasons; commit.

### Task 2: Authentic TRAIN replay and fair runtime
**Files:** `src/mechet/rsi_replay.py`, `scripts/rsi_runtime.py`, tests `test_rsi_replay.py`, `test_rsi_v1_runtime.py`.
**Interfaces:** streaming selected TRAIN source records yield verified `prompt_ids/completion_ids` and source provenance; optional v1 `seed_contract/temperature/rejection_guard` arguments default to unchanged v0 behavior; `per_target_budget` preserves every target.
- [ ] RED exact real-token masks/EOS, source-ID/split rejection, replay mismatch, SHA-independent seeds, repeat-rejection guard and per-target-censor tests.
- [ ] Implement selected512TRAIN reaction replay pool with1–4balanced actions, execute all reference actions to verify prompt/history/state; no 2M-object load. Isolate v1 cache namespace and optional guard; keep v0 defaults.
- [ ] GREEN tests, actual saved-tokenizer parity and source-file audit; commit.

### Task 3: Capped anchored updates and controls
**Files:** extend `src/mechet/rsi_objective.py`; tests `test_rsi_real_anchored_training.py`.
**Interfaces:** derived common update plan, fixed-R0 reference margins, source-balanced DPO groups, EOS-masked token-normalized replay CE; `train_anchored` produces immutable model/provenance/branch-gradient/exposure reports.
- [ ] RED tiny actual Qwen+LoRA CPU tests for DPO/replay gradients, frozen refs, real weight change/save, zero-pair behavior, per-pair cap and paired control work accounting.
- [ ] Implement M1/M2/M3 with common scored examples/update budget, explicitly report active-loss tokens separately from equal scored-token work; <=20steps, <=4exposures/pair, no unbounded wraparound.
- [ ] GREEN complete CPU tests, commit.

### Task 4: Real end-to-end correction run
**Files:** `scripts/run_mechet_rsi_anchored.py`, `scripts/run_taiji_rsi_anchored.sh`, actual resolved YAML/public task config, result report.
**Interfaces:** prior tasks feed S0→within-target forward checks→M1/M2/M3→all-model validation→fresh M3 D1/R2/staticD0 when viable. Frozen original F0 checks with precise cache provenance, default-log heartbeat and separate output.
- [ ] RED orchestration tests for no source overwrite, fresh actor identity, common decoder/seed and full per-target work denominator; implement runnable config/runner.
- [ ] Full pytest and fresh-context whole-branch review; one TDD fix pass for important findings.
- [ ] Verify all source/checkpoint contracts, check current ordinary2GPU resource, submit via proven Tencent/Ceph/default45s stdout launcher. Confirm actual generation/update/evaluation, handle correctable failures.
- [ ] Publish real results/provenance/cases to PR92, never merge. No improvement claim without measured held-out outcomes.
