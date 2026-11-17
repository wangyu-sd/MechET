# PR111/112 complete acceleration implementation and admission status

The earlier deployed HF condition is partial acceleration. It is not the complete PR111 design. The user's instruction authorizes local T4 software checks and stateful replacement of the original two tasks, superseding the design's original no-restart paragraph.


## Current status — 2026-11-18 01:15 CST

Implementation is present; full runtime acceptance is **not complete**. Both native02 instances are END, with **zero new native optimizer updates**. Corrected source actor/optimizer/per-rank RNG and full data order remain D3 step19 / D1 step89. Native03 source is frozen at `9d77c41`; same-task replacement is being prepared, maximum two4-H20 allocations.

D3 passed the standalone H20 BF16 numerical gate (max logprob difference0.2166711), but four-rank startup failed: `TORCHELASTIC_USE_AGENT_STORE=True` leaked into the independent native child, making rank0 a TCPStore client with no server. Child-only environment isolation now clears the inherited distributed/agent namespace; real pinned Torch2.11 TCPStore and actual T4 native inference under inherited torchrun variables pass (T4 max difference0.01003826). This is not yet proof of a repaired four-H20 launch.

D1's first H20 gate failed at0.30258417 against the unchanged0.25 threshold. All16 original outputs/1067 positions remain preserved. H20 recomputation at batch1 was0.1399969 and batch4 was0.2309947; matching original GPU/workflow/outer autocast reproduced0.1399969 with unchanged LoRA parameters. The original discrepancy is **unexplained**, not relabelled PASS. New D1 admission binds the failed report/outputs by SHA and rechecks every old output at both batch1 and batch4, in addition to fresh native validation. Exact teacher vectors now persist even on failure; unchanged frozen failures block automatic re-admission retries.

A separate startup stack audit found eager import of37 optional FLA model architectures. The new scoped import uses installed FLA0.5.2's supported absent-optional-extension path, leaving actual tensor kernels/modules unchanged. H20 eight forward/gradient cases and official Qwen class load passed, optional registry imports0. Total354.2s includes imports/kernels and is **not a matched startup speed comparison**. Source-bound NVML is retained; owned stopped02 compiled-kernel caches are reused on Ceph.

Latest focused repair suite25PASS56.11s (overlaps earlier suites); final read-only review found no critical/important issue. A minor lost-replay-witness finding received1RED then the passing regression. H20 small-workload timing was negative: D3 HF/native median ratio0.4621 at8 requests/4 slots. No H20 full-round speedup, new update or matched-budget quality gain is claimed. Completion requires successful same-task four-rank native startup, original probability receipts, first complete new optimizer checkpoint, and measured useful work/whole-round timing.

Code: [child isolation](../../src/mechet/route_vllm_environment.py), [FLA import](../../src/mechet/route_fla_import.py), [failed-case scorer/witnesses](../../src/mechet/route_probability_admission.py), [admission entry](../../scripts/run_route_native_admission.py). Existing private controller files are available in the audit source mirrors listed below.

## Implemented in the full native candidate

| Requirement | Implementation | Verification / deployment boundary |
|---|---|---|
| P0 exact chemistry reuse | `route_execution_cache.py`: pure FLOW execution, single flight, per-proposal reconstruction and witness provenance; resource errors uncached | Existing chemistry/cache tests, new reconstruction/provenance regressions |
| P0 G4 and credit lanes | `route_refit_runtime.py`, `route_refit_credit.py`: original G4 independent; M1 coverage only after the true receipt owner is harvested with an eligible completed return; explicit credit/exploration budgets | 119 affected tests passed; one unsupported host-Qwen architecture test recorded separately, not counted as passed |
| P1 global CPU/GPU queue | `route_work_broker.py`, `route_global_sampler.py`, `route_distributed_collection.py`: target ownership, bounded request queue, target fairness and estimated prefill/KV admission | Real CPU distributed and exact-recovery tests; native request results retain original IDs/logprobs |
| P2 continuous engine | `route_gpu_backend.py` connects existing native `VllmRouteSampler` to global workers; completed requests refill independently of long neighbors | Controlled-engine regression passed; actual T4 refill checks passed; H20 four-rank repair still pending |
| P2 exact prefix reuse | `route_vllm_cache.py`, `serve_route_vllm.py`: exact token cache, actor/tool/tokenizer/precision salt, quiescent LoRA switch and real cache reset | Actual T4 initial native admission passed probability/prompt/cache/switch checks; max logprob drift 0.0152283 < unchanged 0.25 gate |
| P2 durable receipts and I/O | `route_vllm_backend.py`: original write-ahead requests/completions; fsync group commit; actual bytes/write/fsync metrics | Exact crash recovery and receipt accounting tests; no teacher-forced replacement of native behavior scores |
| P3 learner | Existing B4 objective path plus new `route_reference_cache.py`, used only for frozen reference outputs on exact ID pairs | Numerical loss/KL/gradient/parameter checks; actor logits/gradients uncached |
| Telemetry | `route_vllm_metrics.py`, `route_request_accounting.py`: actual native scheduler/cache/work counters and per-GPU/per-target lane joins | Request accounting 3 RED then 13 affected PASS; request latency sums explicitly are not GPU-kernel timing |

## Earlier native02 launch record (superseded; both instances END)

The old HF instances have stopped at fully completed D3 step19 and D1 step89. Their original task flags have been replaced by the full native candidate, with actor/optimizer/per-rank RNG/full order authenticated and preserved. Both new four-H20 instances are running their H20 BF16 admission gate; they have no new optimizer update yet. Seeded19/89 are continuation start points, not native training results.

The initial T4 test used a frozen earlier source snapshot and four actual requests per backend. It is not final full-candidate admission. Its first timing comparison had asymmetric HF warmup; the raw record is preserved and no speedup is claimed from it. A new test uses independent warmups, alternating timing order, per-phase native metrics and an explicit one-token boundary request to prove refill while uncensored neighbors continue. The boundary request is excluded from complete-action timing.

Native startup took 1140 seconds on the initial cold T4 attempt, including kernel compilation/autotuning and serial shared-storage artifact writes. The full condition now sets the upstream `TRITON_STORE_BINARY_ONLY=1` option to reduce cache artifacts; this is not evidence of steady-state speedup. All caches remain on Ceph, including reuse of this session's own stopped T4 cache.

An H20 BF16 probability/cache/refill gate must pass on the exact final stopped actor before formal native training begins. T4 FP16 checks do not satisfy that gate. The prepared bootstrap records a real H20 software receipt; the training controller verifies source actor, contract, code and precision bindings before starting collection. This gate is currently implementation under verification, not a claimed hardware result.

Both conditions preserve their full data, fixed corrected History89442 reference, reward/recovery/ReFIT semantics, optimizer, per-rank RNG, target order and full validation schedule. D3 TRAIN156723/VALID8250; D1 TRAIN23730/VALID2986. Aggregate allocation remains at most8 H20. Full RL/VALID, public Reference1084/n5-10000, and final TRAIN-only search-to-policy distillation remain required.

## Source availability

New public helper modules and focused tests are in repository-root paths. Existing private runtime controllers and native-server modules are mirrored under `docs/audits/pr111_full_acceleration_20261117/source/` with their original relative paths, and a patch against private base9938c63 records the exact integration. A fresh public checkout alone does not include private data, weights, runtime packages or all historical controller dependencies. Do not interpret an audit source mirror as an independently runnable public training package.

## Private reproducibility evidence

`outputs/diagnostics/pr111_full_acceleration_20261117_01/` contains immutable RED/GREEN logs, actor/source/runtime bindings and each native admission attempt. Raw model prompts/chemistry and private platform initializer credentials are not published. Software checks are distinct from route benchmark or full-training speed results.

### Subsequent T4 attempt02

Actual four-slot/six-request refill passed: the next submission precedes completion of initial long neighbors. Max native/HF logprob difference0.01459074. Balanced four-request timings yielded median HF/native ratio0.8184, so vLLM was slower on this local diagnostic. This is preserved negative speed evidence; no H20 or full-round gain is claimed. Warm-cache native startup302.017s. Source snapshot9a4ef76 excludes later hardware-gate/controller accounting edits, which require final-candidate revalidation. Native metrics snapshots are cumulative within an actor version; do not add them across phases.

## Historical local admission and native02 handoff (2026-11-17 23:20 CST)

Final T4 attempt04 used16 saved requests at8 physical slots, with two queued waves and four alternating, independently warmed backend repeats. HF times[15.2466,17.2709,15.6669,19.4760]s; native[14.1761,17.0674,13.5678,17.1242]s; median ratio1.05423. Max native/teacher-forced logprob difference0.02919769, unchanged0.25 gate. Real10request/8slot refill, exact prompt, cache reuse and actor-switch invalidation passed. This is a local software workload with independently sampled responses, not an H20/full-training throughput or route-quality claim. Earlier4request negative timings remain preserved.

Final combined software suite66PASS71.32s; source-bound NVML packaging regression1RED then46continuationPASS99.68s (overlapping suites). Actual Qwen policy-generation unit passed in pinned TF5.5.0 runtime1PASS4.33s; it had been unsupported by the default host architecture. All generated dependencies/temp/cache data reside on Ceph.

The packaging check discovered the old working environment's explicitly bound NVIDIA pynvml leaf is absent from git archives. The preparer now carries forward that exact SHA before freezing the new inventory. No existing live code or vendor dependency was altered; unused candidate01 is marked DO_NOT_SUBMIT. This packaging-only fix does not change the T4-admitted compute bytes.

| Condition | Same original task flag | New instance | Source checkpoint | New contract |
|---|---|---|---|---|
| D3 | `meteor_mechet_pr109_standard_d_4h20_20261116` | `8b1d89f6a1cfaa4801a1ea0e2de229e3` | step19 | `47fcfbf670a6c166b8d1c99e436f98c62777a823e2cdb18293f886009ed52827` |
| D1 | `meteor_mechet_stage2_native_process_grpo_4h20_zjk_20261115_01` | `8b1d89c6a1cfaaaf01a1ea10ea942910` | step89 | `825912e080a25da69cf4020fa968ec4747c6a2f532ae8e1cf11027b46c652cfa` |

Each requests4 H20, aggregate8. CLI modifications were read back before start; old END receipts and released worker locks were required. Both allocated devices report NVIDIA H20. Read-only monitors distinguish new optimizer updates from seeded progress, and collect five-second device memory/utilization samples; sampled peaks are not exact maxima. Neither monitor automatically submits or restarts a task.

H20 hardware gate remains pending at this timestamp, with both native engines loading. Formal collection starts only after a verified exact-source-actor, contract/code/precision-bound PASS receipt. No native H20 speedup, full-round timing, new update, or matched-budget quality result is claimed yet.
