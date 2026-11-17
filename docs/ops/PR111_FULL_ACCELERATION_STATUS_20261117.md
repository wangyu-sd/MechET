# PR111/112 complete acceleration implementation and admission status

The earlier deployed HF condition is partial acceleration. It is not the complete PR111 design. The user's instruction authorizes local T4 software checks and stateful replacement of the original two tasks, superseding the design's original no-restart paragraph.

## Implemented in the full native candidate

| Requirement | Implementation | Verification / deployment boundary |
|---|---|---|
| P0 exact chemistry reuse | `route_execution_cache.py`: pure FLOW execution, single flight, per-proposal reconstruction and witness provenance; resource errors uncached | Existing chemistry/cache tests, new reconstruction/provenance regressions |
| P0 G4 and credit lanes | `route_refit_runtime.py`, `route_refit_credit.py`: original G4 independent; M1 coverage only after the true receipt owner is harvested with an eligible completed return; explicit credit/exploration budgets | 119 affected tests passed; one unsupported host-Qwen architecture test recorded separately, not counted as passed |
| P1 global CPU/GPU queue | `route_work_broker.py`, `route_global_sampler.py`, `route_distributed_collection.py`: target ownership, bounded request queue, target fairness and estimated prefill/KV admission | Real CPU distributed and exact-recovery tests; native request results retain original IDs/logprobs |
| P2 continuous engine | `route_gpu_backend.py` connects existing native `VllmRouteSampler` to global workers; completed requests refill independently of long neighbors | Controlled-engine regression passed; actual T4 final-candidate refill probe in progress |
| P2 exact prefix reuse | `route_vllm_cache.py`, `serve_route_vllm.py`: exact token cache, actor/tool/tokenizer/precision salt, quiescent LoRA switch and real cache reset | Actual T4 initial native admission passed probability/prompt/cache/switch checks; max logprob drift 0.0152283 < unchanged 0.25 gate |
| P2 durable receipts and I/O | `route_vllm_backend.py`: original write-ahead requests/completions; fsync group commit; actual bytes/write/fsync metrics | Exact crash recovery and receipt accounting tests; no teacher-forced replacement of native behavior scores |
| P3 learner | Existing B4 objective path plus new `route_reference_cache.py`, used only for frozen reference outputs on exact ID pairs | Numerical loss/KL/gradient/parameter checks; actor logits/gradients uncached |
| Telemetry | `route_vllm_metrics.py`, `route_request_accounting.py`: actual native scheduler/cache/work counters and per-GPU/per-target lane joins | Request accounting 3 RED then 13 affected PASS; request latency sums explicitly are not GPU-kernel timing |

## What is still not admitted or deployed

The original task flags continue on the v1 HF runtime while full-native software checks and stateful handoff tooling finish. At the latest read during this update, D3 has completed optimizer step17 and D1 step87. These are actual post-resume updates, not seeded checkpoint counts.

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
