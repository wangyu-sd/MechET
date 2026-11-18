# PR111 acceleration implementation plan

> Execute inline with superpowers:executing-plans. User approved implementation of PR111 and continuation of the original research plan.

**Goal:** Reduce redundant deterministic execution and unused duplicate suffix work; distribute same-version inference across GPUs without altering reward or fabricating behavior receipts.

**Architecture:** Opt-in versioned acceleration configuration. Chemistry cache is distinct from sampled receipts and continuation values. Dynamic target ownership and global model transport keep CPU route ownership while sharing immutable GPU actors. Existing HF and vLLM paths remain fallbacks.

**Spec:** docs/ops/REFIT_PO_ACCELERATION_DESIGN_20261117.md (PR111 b282d7bed11341e66d2426faaae4e2d70dcffd78).

## Global constraints

- Keep current frozen D3 and D1 jobs unchanged; no automatic new GPU job submission. Full protocols only, no unrequested smoke/subset experiments.
- Same corrected Stage-II initialization, dataset/VALID, reward, alpha=.5, M=1/equal-semantic advantages and final TRAIN-only distillation.
- All host data/cache/test outputs on Ceph; unique real receipts and original sampling probabilities survive all optimizations.

## Review focus

- Cache must rebuild per-draw metadata and recheck cycles, not reuse another route's provenance.
- Pruned extra duplicates stay unknown and cannot invalidate original G4 or fabricate branch Q.
- Lease recovery and request replay reject mismatched actor/prompt bindings and double completion.
- Empty learner ranks participate with zero gradients, without changing global normalization.
- Dynamic inference workers stop before learner touches their actor; failures propagate without deadlocks.

### Task 1: Read-only baseline audit and deterministic execution cache

Files: new route_execution_cache.py, route_acceleration.py, audit script and tests; integrate NativeChemistry opt-in. Counters split draws, deterministic executions, cache/inflight reuse. Tests first: concurrent FLOW, different history/raw text, local cycles, resource errors, reconstruction/provenance; run red then green and existing route tests.

### Task 2: Explicit credit/exploration lane and unknown duplicate stop

Files: route_refit_runtime.py, route_refit_credit.py, reward scoring integration, tests. Keep original G4 complete. Preserve sampled duplicate receipt but stop extra credit-only suffix with unknown status, exclude its path from Q/loss, no chemical penalty; replay follows same decision. Tests cover Q/fallback conservation, async/rolling, declarations and censoring.

### Task 3: Dynamic group ownership and cross-GPU request broker

Files: route_work_broker.py, route_distributed_collection.py, training entry integration and tests. Durable per-target claims/results; same-version in-memory RPC for requests/results with single completion, actor binding, bounded admission, GPU worker serving batches from any CPU owner. Recovery consumes durable GPU journals; no token RPC over Ceph. Test ownership, interruption, invalid identities, zero-work ranks and worker failure.

### Task 4: Sampling and learner efficiency

Files: sampling precision/batching helpers and learner options. Preserve per-request RNG/distribution; remove redundant original-distribution scans when enabled, length bucket HF admissions, reuse existing validated async vLLM interface only behind its gate; golden HF fallback. Length-aware learner microbatch option with full real tensor loss/gradient parity; optional reference cache only if useful. No claim of continuous HF decoding without real independent KV/EOS implementation.

### Task 5: Integration, review and preparation for the next frozen condition

Run focused suites, CPU distributed routing/learner tests, read-only saved-receipt classification, source/contract/hash checks. Independent whole-change review; fix important findings with regressions. Publish implementation/evidence and current deployment limits to PR. Prepare separate full-D3 accelerated contract/queue plan; preserve existing jobs and research sequence. Performance acceptance requires measured work/wait reductions, not GPU-util alone.
