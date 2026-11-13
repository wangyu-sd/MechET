# Eight-H20 Qwen3-8B RSI implementation plan

> Execute inline with superpowers:executing-plans. User explicitly authorized continuous execution and PR upload; do not request repeated approval.

**Goal:** Queue corrected8B training after full CPU cache verification; provide usable evidence-backed route memory and explicit readiness for downstream data/adaptation/RL.
**Spec:** docs/ops/QWEN3_8B_RSI_8H20_20261113.md
**Architecture:** retain immutable four-H20 condition; add an eight-H20 condition sharing the four-shard model-bound cache. Integrate CPU memory at the episode observation boundary and bind exact rendered memory input to records. Independently audited replay produces training evidence and mixed-data manifests.

## Tasks

- [x] Publish approved design plus reviewed first-delivery source patch to a dedicated draft PR.
- [x] RED→GREEN tests: eight-H20 config, global64, unchanged four-H20 condition, cache4-shards, resume8-rank RNG. Implement explicit eight-card launcher/config.
- [x] RED→GREEN tests: live memory input preserves history/current-state precedence; immutable per-episode memory; snapshot/restore and training/inference parity; no cross-scope stale handles; actual evidence required.
- [x] Implement replay/curation entry point with hash-bound records, TRAIN-only receipts, real rejection/recovery checks and explicit eligible/unsupported counts. Verify deterministic idempotence and censored exclusions.
- [ ] Prepare immutable Taiji eight-H20 SFT source/runtime/contract; submit only after full cache verification. Monitor startup and repair reproducible runtime failures within protocol.
- [x] Prepare downstream stage dependencies with real readiness checks. No unbound actor hash, fake precomputed data or silently substituted denominator can launch memory adaptation/RL.
- [ ] Run focused regressions and final independent review; commit owned files and update PR with evidence and exact queued/running/not-ready states.

## Review focus

Eight-card changes must not weaken existing0.8B or four-card model guards. Reusing a cache cannot waive model/tokenizer/source bindings. A memory receipt must bind the exact content, scope and completed episode, not just mention a successful run. Memory cannot inject chat delimiters or current atom commands. Only eligible TRAIN episodes may build persistent memory; heldout data and physical resource failures must be excluded. A queued task is not a verified training update.


Ruling: maintain the existing4CPU cache job unchanged and share its four verified shards with8 learners — model/data/protocol are identical; a mismatch still fails verification.
Ruling: memory transfers only three typed error-recovery advice kinds initially — this makes applicability auditable; broader semantic generalization remains future work, not an achieved result.
Ruling: downstream stages remain explicit dependencies until new8B weights, replay counts and backend binding exist — avoids launching under an old actor; costs waiting for the real upstream run.
Final review: four Important findings plus unstarted snapshot API defect reproduced and fixed; focused52PASS and subsequent11PASS overlap earlier counts. Final expanded affected run92PASS88.45s; full-project attempt timed out180s/exit124, no full-suite claim.
