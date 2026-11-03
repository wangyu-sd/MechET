# PR96 dual-track four-GPU EARHO optimization

> **For agentic workers:** Use superpowers:executing-plans inline; final fresh-context review required.

**Goal:** Optimize and submit separate FlowER and mech31k Stage-III jobs on one ordinary four-H20 host each.

**Architecture:** Preserve the existing actor-only adaptive EARHO and each track's immutable Stage-II source/parent. Replace per-rank full JSON materialization with a shared authenticated byte-offset index and per-round reference replay; coalesce gradient collectives without changing global group normalization. Make GPU count and checkpoint schedule explicit rather than hardcoded eight/10137.

**Tech Stack:** Existing PyTorch distributed/Transformers5.5/PEFT0.18.1/native CUDA11 causal-conv runtime; SQLite offsets; Tencent taiji6.

**Spec:** Latest user request, PROJECT_MEMORY.md, ACTIVE_ICLR_STATUS.md and the already-approved 2026-10-23-pr96-earho-stage3.md scientific contract. No further algorithm or model experiment design is requested.

## Global Constraints

- FlowER accepted source-arrow TRAIN243753, original VALID2890; mech31k accepted TRAIN10137, original VALID1319. Not unqualified full endpoint data.
- FlowER History89442 SHA2c98a9bff3afeaffbf2ee2733885cccc68e936f9310fd9f46ba4ee2211e5f453 with c42c94b source; mech31k History660 SHAcd5f2f9dbdc883674be2f87781452366a3fb2c01c7c930482e0b5e4b3dab49ca with731af13 source.
- K2, adaptive H2..12,25%root rehearsal,128roots/round,two optimizer passes, global batch64 and reward/loss unchanged. No TEST reads or diagnostic actor.
- Four actual GPUs per job; Ceph-only data/cache/results; no new model smoke; all raw I/O retained, default stdout heartbeat45s and meteor prefix.

## Review Focus

- Wrong cross-track actor/source accepted by a generalized contract.
- Partial final round or missing rank RNG accepted by checkpoint recovery.
- Noncontiguous history IDs silently dropped or replayed out of order.
- Lazy replay retaining previous-round states and growing memory with TRAIN size.
- Bucketed collectives changing unequal-rank/missing-gradient normalization.

### Task 1: Implement verified four-rank, two-track runtime

**Files:** scripts/pr96_earho_contract.py, scripts/pr96_earho_storage.py, scripts/train_pr96_earho.py, scripts/prepare_pr96_earho_submission.py, scripts/taiji_pr96_earho_bootstrap.sh, tests/test_pr96_earho*.py.

**Interfaces:** explicit track profile; verify_checkpoint(...,cfg); IndexedJsonl(path,index,kind) supports IDs and selected ordered rows; synchronized gradients remain exact group mean.

- [x] Write failing real behavior tests for track/world validation, four-rank last-round checkpoint, lazy index order/duplicates, missing gradients and bucketed exactness.
- [x] Run targeted tests to observe failures; implement minimal changes and rerun green.
- [x] Authenticate both parents using their own immutable source; run full software suite, shell checks and final fresh review. Commit scoped files.

### Task 2: Freeze and deploy

**Files:** Ceph contracts/receipts and current memory/status entries.

**Interfaces:** Four-rank bootstrap, track-specific frozen archives/contracts, successful own H20 donor private init.

- [ ] Verify old eight-H20 EARHO stopped; preserve outputs/checkpoints and unrelated tasks.
- [ ] Freeze all inputs and source, submit two ordinary four-H20 tasks.
- [ ] Check real POD/Ceph/fourH20/default stdout/model generation/updates; distinguish queue/startup from training. Record exact IDs and remaining verification.
