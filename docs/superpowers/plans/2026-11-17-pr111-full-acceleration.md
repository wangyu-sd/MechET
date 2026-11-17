# PR111 full acceleration implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox syntax for tracking.

**Goal:** Complete and verify every mandatory PR111 acceleration requirement, including a real continuous generation engine, then continue the original D3/D1 full jobs from authenticated completed checkpoints.

**Architecture:** Reuse the existing isolated vLLM native-token server and its asynchronous per-request Futures behind the global same-actor broker. Keep CPU chemistry/credit ownership and the existing learner. Add version-bound prefix caching, explicit credit/exploration accounting, fair capacity-aware dispatch, and reconciled physical-work telemetry. The old HF condition stays reproducible.

**Tech stack:** pinned vLLM0.24.0+cu129/Torch2.11.0+cu129/Transformers5.5.4 engine; existing Torch2.6/Transformers5.5.0 BF16 learner; four H20 per condition.

**Spec:** `docs/ops/REFIT_PO_ACCELERATION_DESIGN_20261117.md`, approved PR111 b282d7bed11341e66d2426faaae4e2d70dcffd78. PR112 is its implementation branch.

## Global constraints

- Preserve corrected History89442 fixed reference, each current trained actor/optimizer/per-rank RNG/full order, D3 TRAIN156723/VALID8250 and D1 TRAIN23730/VALID2986.
- Preserve original G4 independence, D3 first-rejection/ReFIT credit formulas, D1 eight recoveries/no ReFIT, all reward/endpoint/test-isolation contracts, full VALID and final TRAIN-only distillation.
- One frozen behavior version per collection round; finish/reconcile all selected target groups and engine requests before learner update. Source outputs remain unchanged.
- Distinct sampled requests retain their own actual generated IDs and original probabilities. No synthetic independent draws, cached rewards, cross-history suffix assumptions, or censored-to-failure coercion.
- Original design's no-restart paragraph was superseded by the user's explicit instruction to validate locally on T4 and replace original D3/D1 task runtimes from complete checkpoints. User now explicitly requires the entire acceleration plan; do not seek redundant plan/approval confirmation.
- Maximum aggregate eight H20. No extra training/evaluation population, smoke/pilot or benchmark substitution. Authorized local T4 software/short validation is distinct from production results. All artifacts, test basetemps, caches and runtime writes on Ceph; frozen running code never edited.
- Use existing isolated WORK. Unrelated dirty MCTS/memory files remain untouched except append-only operational notes. Private initializer/tokens never published.

## Review focus

1. Engine request completes during refill while a long neighbor remains active: no lost future, duplicate request or premature engine/learner overlap (Task1).
2. Restart between write-ahead request and returned native receipt: reuse actual durable IDs/logprobs, mark unfinished physical draw explicitly, never recompute or silently redraw (Task1).
3. Adapter changes while exact prompt repeats: drain, independently verify LoRA SHA, reset KV, unique actor/tool/tokenizer/precision salt; stale prefix forbidden (Task2).
4. First accepted continuation later censors: accepted edge alone never satisfies declared M; all four originals continue even on duplicates (Task3).
5. A long prompt among many short requests and a zero-work owner: bounded admission plus target fairness, no starvation or objective denominator change (Task4).

## Task 1: Native continuous engine in the global runtime

**Files:** modify `src/mechet/route_acceleration.py`, `route_distributed_collection.py`, `route_global_sampler.py`, `route_vllm_backend.py`; tests `tests/test_route_global_native.py`, existing native/global/recovery suites.

**Interfaces:** `full_acceleration(refit: bool) -> dict`; full version `pr111_full_acceleration_v2`, backend vLLM, sampler `vllm_native_v1`. Add a shared native/HF GPU-worker factory and backend-aware journal recovery. Keep the v1 HF contract unchanged. Native worker owns one VllmRouteSampler per GPU/version, binds the checkpoint once, submits independent Futures, drains and closes the sampler before collection-to-learning transition. Native telemetry/journal remain engine-original; no teacher-forced replacement of behavior scores.

- [x] Add RED tests `test_global_native_eos_refills_before_long_neighbor_finishes`, `test_global_native_recovery_reuses_exact_original_receipt`, native source-version rejection, worker failure propagation and adapter bind exactly once per version.
- [x] Run failing tests with isolated Ceph basetemp; distinguish missing implementation from fixture/import errors.
- [x] Wire the existing native transport through the global collector; replace unconditional HF-only restriction with explicit v1/v2 validation.
- [x] Run affected global/native/distributed tests, commit and ledger evidence.

## Task 2: Exact prefix reuse and engine telemetry

**Files:** modify `scripts/serve_route_vllm.py`, `src/mechet/route_vllm_protocol.py`, `route_vllm_backend.py`; create focused `route_vllm_metrics.py` if needed; tests native prefix/metrics/protocol.

**Interfaces:** versioned `prefix_namespace(cfg, behavior_version)` includes adapter version, frozen tokenizer/tool source identity, precision and model identity. Exact IDs remain engine cache keys. Native engine kwargs enable prefix caching, Mamba `align` and chunked prefill; LoRA switch requires drain plus successful `await engine.reset_prefix_cache()` and unique LoRA name. Native completion contains `num_cached_tokens` and timing/provenance metadata. StatLogger records actual running/waiting requests, KV usage, prefill/decode work and finished-request timings; do not infer GPU occupancy from submitted Futures. Aggregate per GPU/target/round with durable transactions.

- [x] RED tests exact-prefix/actor/tool/precision invalidation, failed cache reset, active request version switch refusal, stats reconciliation and malformed native statistics.
- [x] Implement the pinned engine APIs and native cache/timing receipt fields; keep probability checks strict and generation unmasked.
- [x] Verify native protocol and simulated engine boundary tests; commit and ledger.

## Task 3: Complete P0 proposal/continuation lanes and cache provenance

**Files:** `route_refit_runtime.py`, `route_execution_cache.py`, `train_external_route_rl.py` integration; focused lane/cache tests.

**Interfaces:** `RefitCollection(..., lane_config=None)`; v2 lanes declare `credit_forks_per_tree=16`, `exploration_forks_per_tree=0` initially, always bounded by unchanged max16 total forks. Credit covers first declared eligible real M1 result per exact state/semantic FLOW; unsuccessful/censored pending continuation does not satisfy M. Explicit extra exploration is separately versioned/budgeted and never adds M1 sibling votes. Pack lane proposal/continuation/skipped/outcome counters and provenance. Maintain v1 old replay behavior. Cache metrics include producer/witness provenance and reconstruction-failure count while retaining receipt-specific metadata and local guards.

- [x] RED tests all four G4 duplicates independent, absent/unfinished/censored M1 not pruned, completed eligible M1 duplicate unknown skip, explicit lane budget separation and replay identity, no hidden-reward selection, cache provenance/reconstruction failure.
- [x] Implement opt-in v2 lanes and counters without altering reward/credit formulas or old replay semantics.
- [x] Run real chemistry/cache/ReFIT credit/replay suites; commit and ledger.

## Task 4: Fair workload and capacity-aware dispatch

**Files:** `route_work_broker.py`, `route_global_sampler.py`, native distributed wiring; broker/admission tests.

**Interfaces:** `RemoteSampler.submit` sends validated prompt length/independent output budget and CPU target owner. WorkBroker accepts versioned optional dispatch settings; native v2 `take` admits bounded estimated prompt/decode work, alternates targets fairly and records queue/latency/ownership counters. v1 remains FIFO for exact historical replay. Native engine applies actual KV/prefill ceilings. Oversized individual requests become explicit capacity outcomes, never starve behind short work or silently change action distributions.

- [x] RED tests unequal target queues, long prompt vs short, padded/prefill capacity, bounded pending admission, request starvation and restart ownership, zero-target rank synchronization.
- [x] Implement fair selection and useful-wait telemetry; retain strict actor/prompt binding and request IDs.
- [x] Run multiprocess broker/global/gradient denominator tests; commit and ledger.

## Task 5: P3 numerical verification and optional reference cache

**Files:** learner `external_route_rl.py`; optional new `route_reference_cache.py`; learner/reference tests and runtime contract settings.

**Interfaces:** immutable bounded reference cache keyed on exact prompt/response IDs, frozen reference SHA, tokenizer/precision/mask; stores only reference scores detached on CPU. Actor logits/gradients are never cached. Feature remains explicitly configurable. Batch4 retains fixed N*G*1024 normalization and all real receipt multiplicities.

- [x] RED cache identity/invalidation and opposing-advantage numerical tests; implement if it yields reuse without unbounded transfer/storage cost.
- [x] Re-run real objective/KL/loss/gradient/parameter-delta B1/B4 parity and zero-work rank tests. Optional reference LRU may remain disabled with explicit evidence, per spec; never label enabled when not deployed.
- [x] Commit numerical evidence and telemetry reconciliation.

## Task 6: Actual engine admission, measured work and formal continuation

**Files:** new local native admission script; extend `route_acceleration_continuation.py` / preparation CLI and tests for native receipt-bound handoff; operator receipts; public PR112 source/evidence docs.

- [x] Use existing audited native runtime. Add explicit software-only T4 FP16 profile with small KV and Triton attention; formal engine stays BF16. No generic tolerance relaxation; locate and fix any logprob mismatch by source/evidence.
- [ ] Run real latest-actor prompt/tool parity, native sampled probabilities vs teacher forcing, LoRA effect and version change, mixed EOS refill, repeated exact prefix cache-hit proof, restart receipts and same-workload latency/complete-action throughput versus golden HF. Report first failures and exact numerical tolerances. Local results are not H20 results.
- [ ] Run affected CPU/tensor suites and independent whole-branch review; fix material findings and verify.
- [ ] Prepare concrete new native runtime/contracts and source-migration hashes while original HF jobs continue. Check completed source checkpoints and ownership, then use the already authorized stop/modify/readback/start handoff with total8H20. Local admission does not replace H20 BF16 probability validation; run that on owned allocated resources before declaring native admitted.
- [ ] Verify actual same-task four-rank resume, native process and native probability receipts, useful prefill/decode/cache/queue telemetry, first new completed learner update. Record actual whole-round/workload comparison and separate matched-budget quality from software speed. Preserve full denominators and final distillation sequence.
- [ ] Update PR111/112 scope/status and PROJECT_MEMORY/ACTIVE_ICLR_STATUS with code/tests/deployment evidence. Final answer must list any unmet requirement plainly; partial preparation is not full acceleration completion.

## Execution ledger

Base9938c63. Read-only full-spec and runtime audits completed before implementation. Existing design and explicit execution request are approval for this work; no new design menu. Prefer existing native engine (spec Choice1); retain golden HF. Tests always use a new owned Ceph basetemp to avoid common-root cleanup. Each task's RED/GREEN evidence and deviations are written alongside its commit.

Task1 verification:3 RED missing-feature failures (task1-red.log), then48/48 GREEN30.31s (task1-green.log). Native test uses real broker/transport/journal boundaries with controlled external engine Futures, not GPU inference; no native production claim. Existing per-round checkpoint pointer update confirmed in train_external_route_rl.py:690.

Task2 verification:10 initial RED cases; first pass54PASS/1FAIL exposed empty-version counters, fixed. Missing observed-cache-counter guard reproduced RED, then fixed. Legacy saved SimpleNamespace request compatibility reproduced RED(asdict TypeError), fixed without changing DecisionRequest journal identity. Final affected suite59PASS34.58s (task2-final.log). Native scheduler stats are real engine observation interfaces, not yet GPU-validated; request timing sums explicitly not GPU kernel time.

Task4 verification:4RED constructor/feature failures, then affected26PASS56.50s (task4-green02.log), including actual two-process Gloo chemistry/gradient integration. Initial22PASS/2FAIL caused by old StubBackend fixture omitting newly owned writer lifecycle; fixture now retains actual ReceiptBatch/execution_writer. Extra recovery/context-neighbor regressions included. v1FIFO remains; v2 rotates targets and admits prefill/padded/reserved-token estimates, idling higher-estimated-load GPU until lighter replica can claim.
Ruling: delegate independent Task3 P0 code domain to one standard implementation agent while controller runs immutable nativeengineT4 admission and owns Tasks2/4/5/6; no concurrent edits to shared sourcefiles. Existing approved PR design is executionauthorization. CPUtest basetemps separate andowned, frozen running code unchanged.

Task3 complete71e43e5:119PASS/1 unsupported hostQwen architecture case deselected; fullreport outputs/diagnostics/pr111_full_acceleration_20261117_01/p0_task_report.md. Independent P0 spec+quality review clean (no new tests/models by reviewer). v1 retained; owned uncensored terminal failures count M1, pending/censored/copiedprefix do not.
Task5 local affected16PASS18.91s, CPU gradient/parameter-delta/opposing advantages and referenceKL parity. One hostCUDA-driver warning onCPUbackward retained; no inferencefailure or GPU claim. Refcache only immutableCPUreferenceoutputs, keys fixedreference+tokenizer/precision/mask; actorforward/gradients remain uncached.
Task6 initial actual T4 engine33b107f PASSED at21:52: sourceD3step15 andD1step85, max native/HF FP16 LPdiff0.0152283, exactprompt/completeactions/cachehit/versionreset checksPASS. Enginecoldstart1140s duemanynativekernelcompiles/autotunes+serializedCephwrites; officialTRITON_STORE_BINARY_ONLY1 implemented onlyfornewfullnativechild afterRED. First4request timing ratiomedian1.09 isNOTa speedupclaim: HFinitialmeasurement notindependentlywarmed, responsesindependentlysampled. Keep rawPASS receiptimmutable; rerun balancedwarmup/order and actual stop/refillconcurrency probe fromfinalcandidate beforepromotion. Initial T4 snapshot excludes newer P0/fair/refcache modules; notfullfinaladmission.

Task6 accounting:3RED then13PASS9.70s; broker joins harvested receipt labels without consulting reward or influencing sampling. Copied prefixes retain original owners; native group-commit reports actual bytes, rows, write/fsync seconds and barriers. Diagnostic v2 uses physical single-GPU fair dispatch, independent HF warmup, alternating timing order, per-phase actual engine metrics and a separate one-token EOS/resource-boundary refill probe; all raw timings retained. Existing user-owned stopped T4 cache can be reused explicitly, never other sessions' files.

Task6 native handoff guard:2RED then44PASS98.88s continuation suite; combined affected46PASS79.92s including learner/distributed paths. Added exact positive hardware receipt/code checks and credit accounting:3counterRED then37PASS24.02s. Existing hostCUDA warning retained, CPU tests successful. Bootstrap H20 gate and stateful native preparation implemented but not yet platform deployed or hardware admitted. Old VALID boundary transfer guard remains deliberately explicit; must stop latest source before first VALID boundary or implement provenance support, never drop completed progress.
Actual T4 attempt02 PASS: warm-cache startup302.017s, maxLPdiff0.01459074, real4-slot/6-request refill, actor switch/cache checks passed. Four balanced warmup/order repeats gave HF[5.8609,5.8269,6.5313,10.8701]s/native[8.4070,5.9399,6.7344,10.8872]s, medianratio0.8184 (slower). This is valid negative speed evidence, not acceleration success; retain original. Some extra native kernels JIT within first timed repeat; final candidate will exercise8request concurrency with all results retained. Stage engine counters are cumulative within actor version, not disjoint phase counts.

Final whole-branch review: noCritical; Important recovered finished groups lost lane/target labels, reproduced1RED then11PASS, fixed by joining authenticated scored-group reports after seeding native receipts. Minor cache witness mapping was memory-only, reproduced1RED then24affectedPASS17.75s; new round-barrier provenance file identifies producer/witness while original execution journals remain crash authority. Whole review otherwise sound native lifecycle/probability/prefix/reference/admission/checkpoint boundaries.

Ruling: benchmark a backlog of two request waves at fixed physical concurrency, because the earlier complete-action timing had no waiting request to refill and could not measure the continuous-engine benefit. New final T4 uses16 saved prompts/8 slots, full independent outputs, both backends equally warmed, four alternating repeats. It remains a software workload, not full RL or route evaluation. All previous four-request negative evidence is retained.

Final T4 attempt04 (2917b6f):16complete savedrequests/8slots/twowaves,4alternatingrepeats, numerical/cache/switch/refillPASS maxLPdiff.02919769; HF[15.2466,17.2709,15.6669,19.4760]s/native[14.1761,17.0674,13.5678,17.1242]s median1.05423. Startup209.277s. Finalcombined66PASS71.32s(overlaps other suites). Originalv1 sources stoppedEND, latestD3step19/D1step89, immutable payloads retained.
Preparation found a concrete historical dependency gap before any native submission: git archives omit the bound scripts/_nvml_runtime/pynvml.py leaf, and the original source hashes explicitly bind it. Earlier project history records actual missingNVML failure. Add source-SHA-checked vendor-leaf carry-forward before computing new frozen inventory/archive;1RED then46continuationPASS99.68s. D3candidate01STATIC_PASS butneverstart; superseded by02packaging with this required dependency, not a model/algorithm change. T4 validated compute bytes unchanged by packaging-only preparer delta.
