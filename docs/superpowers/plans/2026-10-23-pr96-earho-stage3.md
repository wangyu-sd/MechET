# PR96 mech31k EARHO Stage-III Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans; execute inline as requested by the user.

**Goal:** Submit one ordinary eight-H20 full accepted-TRAIN EARHO task from the unchanged PR96 History checkpoint, and verify the existing accelerated FlowER replacement.

**Architecture:** Preserve the frozen 731af13 Qwen3.5 language model, executor, public-root construction, XML tool template and compressed history. A separate entry adds repeated current-policy product-start diagnosis, independently replayed first-divergence resets, two stochastic first-action alternatives, chemically pooled successor support, bounded adaptive continuation, success-gated clipped actor updates, fixed-parent KL and explicitly accounted fallback replay. No graph actor or separately learned value critic. Every accepted TRAIN reaction is visited; all original VALID reactions remain the evaluation denominator.

**Tech Stack:** PyTorch distributed, native Transformers5.5/PEFT0.18.1 Qwen3.5-0.8B, RDKit2026.3.4, Tencent taiji6, the verified CUDA11 native-convolution runtime.

**Spec:** User requests in this conversation plus `docs/EXECUTION_ANCHORED_RECEDING_HORIZON_OPTIMIZATION.md` in the historical MechET-pr83-greedy checkout (original frontier/horizon scientific contract); PR96 parent/data contracts govern model-facing observations. This is not the historical critic-enabled Qwen3-8B runner and not the single-round diagnostic.

## Global Constraints

- Parent History660 adapter SHA cd5f2f9dbdc883674be2f87781452366a3fb2c01c7c930482e0b5e4b3dab49ca; never use diagnostic TEST-trained adapters.
- Accepted TRAIN10137; quarantined15 of original trace-view10152 explicitly retained in source accounting. Not full endpoint24959.
- Original trace-view VALID1319, no TEST source loaded, no reference future in actor generation or stopping.
- Ceph for all data/caches/results; immutable source/runtime/config SHA bindings.
- One normal8H20 task, meteor prefix, proven Ceph init, default unbuffered POD stdout/stderr plus45s heartbeat. Do not stop FlowER or unrelated tasks.
- Local unit/static/distributed software tests only; no new unrequested model smoke experiment.

## Review Focus

- Different numbers of supported actions across ranks must not deadlock or bias gradients, including the last partial batch.
- Later rounds must collect from newly updated actor weights, but KL always references original Stage-II.
- Unsupported/truncated continuation is censored, not a claimed terminal error or a positive endpoint.
- A recovered endpoint overrides intermediate reference differences; solved episodes do not create false correction frontiers.
- Checkpoints must authenticate optimizer/RNG/cursor/config and never silently resume another actor or source.

### Task 1: Implement and test the compatible EARHO runner

**Files:** Create `scripts/pr96_earho_contract.py`, `scripts/train_pr96_earho.py`, `tests/test_pr96_earho.py`, and its bootstrap/config.

**Interfaces:** Contract module supplies TRAIN/parent guards, all-ID scheduling, successor-class advantages and horizon update; runner supplies causal token loss, distributed gradient synchronization, collection, update and complete-round checkpoints. The immutable original source supplies executor and native model functions.

- [ ] Write behavioral tests for source/parent rejection, full scheduling, no-positive gating, successor pooling, horizon promotion, endpoint-over-divergence priority, separate behavior/reference likelihoods, CPU device routing and unequal per-rank gradients.
- [ ] Run these tests and observe expected missing-feature failures.
- [ ] Implement minimal compatible runner. Each round collects current-policy product-start episodes then branches at actual first divergences; 25% fixed full-product rehearsal; H2 initially and increases only with effective verified support, max12; K2 unchanged. Cover every TRAIN root once, 128 roots per round, two optimization passes, global batch64. Explicitly distinguish actual PG from fallback replay.
- [ ] Run targeted tests and the whole available suite; record failures without disguising unavailable runtime/data as success.
- [ ] Freeze source/config/runtime and commit only relevant new files.

### Task 2: Submit and verify

**Files:** New Ceph submission contract/receipt; update execution status only.

**Interfaces:** Bootstrap consumes authenticated entry/source/runtime archives, parent/source bindings and the requested ordinary GPU allocation.

- [ ] Verify current FlowER `_04` is the already-deployed accelerated c42c94b successor, with updates beyond29814 and saved checkpoints; do not replace it again.
- [ ] Fresh whole-branch read-only review; fix important findings with RED→GREEN tests.
- [ ] Query ordinary resources; submit exactly one8H20 EARHO task using successful donor Ceph init and Tencent taiji6.
- [ ] Verify POD/Ceph/eight actual H20/default log pipes and actual collection/update, or report the real scheduler/runtime blocker rather than submission success.
- [ ] Save receipt, exact algorithm/source counts/parent SHA, live progress and checkpoint locations; do not claim endpoint gains before held-out evaluation.
