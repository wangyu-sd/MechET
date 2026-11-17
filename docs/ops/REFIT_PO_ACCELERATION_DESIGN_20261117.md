# Retro-E ReFIT-PO: Efficient Inference, Branch Deduplication and GPU Scheduling
**Implementation design / Codex handoff — 2026-11-17 — DOCS ONLY**

Baseline: PR #110 head \`e3adb319ee881ac8d919593d12a1066383086eb2\`. This document describes a **future separately frozen** accelerated run. The formal Standard-D 4×H20 job currently running must not be stopped, edited, reconfigured, or resumed with new code/contract. No FlowER forward teacher, RetroChimera, new Reward, change in Stage-II initialization or new RL objective is part of acceleration.

## 0. Measured bottlenecks and priority

Source evidence:
- \`docs/ops/PR109_D3_REFIT_PROGRESS_20261117.md\`
- \`docs/ops/PR110_DUPLICATES_AND_GPU_20261117.md\`
- \`docs/audits/pr109_refit_20261116/acceleration_20261117.json\`
- Actual \`src/mechet/route_refit.py\`, \`src/mechet/route_refit_runtime.py\`, audit-source mirrors \`route_policy_sampling.py\`, \`route_async_collection.py\`, \`train_external_route_rl.py\`, \`external_route_rl.py\`.

Last audited completed TRAIN round: 64 target source rows, 256 original G4 reports + 3,044 extra ReFIT suffixes, 63,925 physically generated tool decisions (5,419 from original G4), 384 valid vs 1,538 masked same-state conditional groups, 92.29 min wall-clock; slowest collection 81.88 min, generation 69.28 min, learner 9.83 min. Exact current search settings: \`max_nodes_per_tree=8\`, \`max_expansions=2\`, \`max_duplicates=2\`, \`max_forks_per_tree=16\`, \`matched_continuations=1\`; generation per rank ~54–56 requests/batch, learner microbatch=1. Other GPUs can be stalled in \`dist.all_gather_object\` while a rank finishes its assigned targets. GPU-util=100% plus memory-util=0% in a short sample on finished ranks is consistent with NCCL polling, NOT proof of useful high compute utilization.

**Priority order**:
P0. Separate physical computation from reward/behavior receipts and eliminate provably redundant deterministic computation / excess unneeded suffix exploration.
P1. Dynamic *same-policy-version* work distribution across the four GPUs (coarse target-group work stealing first, global model-request queue to cure tails).
P2. Continuous refill inference, prefix KV reuse, vectorized sampling, batched journal.
P3. Larger training minibatches, optional reference-logprob cache, measurable loss-equivalence.
Do not claim any speedup before measuring. Avoid just setting a smaller max_forks value and calling that an engineering acceleration: changing search budget is an algorithmic condition, requiring explicitly separate matched-budget comparison.

## 1. Nonnegotiable separation: compute reuse != reward merging

One **canonical chemical edge** can have multiple **independent sampled action receipts** and multiple distinct downstream **stochastic continuation results**. Do not merge these three ledgers.

\`\`\`text
DecisionNode (exact conditioning, tree/occurrence/history/version)
  ├─ CanonicalSemanticEdge (same actual executor input+semantic action)
  │    ├─ deterministic execution result / error -> SHAREABLE CACHE
  │    ├─ ProposalReceipt#1 (request_id, raw tokens, old logprobs, policy version)
  │    ├─ ProposalReceipt#2 (different REAL draw and its own logprobs)
  │    └─ Continuation#M (real sampled suffix, full route result, own reward/RTG)
  └─ Next distinct semantic edge ...
\`\`\`

- **Executor computation**: share for identical relevant state + grounded action + exact deterministic executor/endpoint/stock/material/route-guard contracts; reconstruct the current receipt's individual action metadata and independently run any route-dependent FINISH/ancestor checks. NEVER blindly clone a cached child object that carries someone else's sampled score, token length, visited set or action history.
- **Policy generation**: each distinct request is a real draw from the current behavior policy. Do NOT synthesize outputs, copy old sampled tokens/logprobs to a new request, or claim two same-semantic but different raw histories have the same future model conditioning.
- **Reward**: route outcome, progress, failure rollback, censoring and RTG are per sampled continuation; a parent edge cache must not store or reuse \`reward\`, \`advantage\` or \`Q\` as an intrinsic chemical property. Failed sibling ≠ failed parent or other sibling.
- **Branch estimator**: \`M=1\` uses the first **declared eligible real continuation** per distinct semantic FLOW action for Q; duplicates do not add votes in sibling baseline. Match the existing exact-conditioning and censor rules. No PUCT visit count treated as a policy-probability denominator.
- **Optimizer**: preserve all eligible **unique actual** receipts; existing \`unique_credit\` and full key compaction use multiplicity and one action-level Advantage per real sampled record. Shared prefix enters loss once. Different positive/negative advantages or behavior probs must NEVER be compacted as if identical samples.

## 2. P0A — accepted-successor deterministic cache, including concurrent in-flight ownership

Current \`ObservedEpisode.accept()\` executes before \`RefitCollection._completed()\` registers the semantic edge. Thus accepted duplicates still execute and produce full suffixes.

Introduce a cache at the chemistry transition layer that is consulted **before the expensive deterministic step**. It must be separate from the search/credit ledgers.

\`\`\`python
ExactExecutionKey = sha256({
  "state_before_mapped": ...,
  "grounded_semantic_action": ...,
  "current_reaction_initial_state": ...,
  "executor_sha": ...,
  "endpoint_contract": ...,
  "material_contract": ...,
  "route_guard_context_sha": ...  # only where route checks are cached
})
ExecutionCacheEntry = {
  "status": "IN_FLIGHT|ACCEPTED|REJECTED|RESOURCE_UNCACHEABLE",
  "chemical_successor": ...,
  "deterministic_error": ...,
  "witness_sha": ...,
  "producer_receipt_id": ...
}
\`\`\`

Use per-key single-flight ownership/futures: one caller executes the real chemistry; concurrent identical proposals subscribe to its result. In-flight deterministic failures are cacheable only with exact matching context. Resource errors/infrastructure failures must never be memoized as chemical failure. Rebuild current sampled receipt's own action metadata/history; recheck local visited, reaction/provenance and route-specific semantics after applying a cached pure chemical successor. On replay/cache hits verify reconstructed state and deterministic receipt equivalence.

A semantic key compiled from ephemeral Axx/Bxx must be grounded under the exact current mapped state; never assume aliases remain stable across observations. FINISH route commitment is context-dependent; first version may cache only accepted FLOW/IMPORT pure chemistry and leave FINISH boundary checks uncached.

Required counters: \`deterministic_execute_calls\`, \`accepted_execution_cache_hits\`, \`rejected_execution_cache_hits\`, \`inflight_waiters\`, \`cache_reconstruction_failures\`. Execution cost savings ≠ new model tokens savings.

## 3. P0B — stop wasting extra ReFIT suffixes without breaking real samples

The current search path mixes two goals:
- **Conditional-credit data**: M=1 matched continuation for each distinct sibling action.
- **Extra exploratory search**: more routes/stock closures, including duplicate accepted actions with no additional Q baseline vote.

Implement a two-lane *explicit allocation protocol*, configurable in a new frozen condition:
- \`credit_lane\`: complete only the declared M continuations for distinct semantic FLOW siblings under one same-state group; full original G4 episodes are ALWAYS retained unmodified.
- \`exploration_lane\`: remaining tree search budget; report its actual additional source requests, distinct actionable successors and incremental solved targets separately. Search expansions can be useful for final distillation even if not for current GRPO, but must not be miscounted as independent M=1 Q supervision.

For **extra ReFIT-only duplicate semantic draws** when an accepted edge has already reached its declared M and another sibling continuation would supply no new estimator sample:
- keep the new real proposal receipt and behavior token/logprobs, plus \`duplicate_pruned_no_continuation\` status;
- **do not generate a new stochastic suffix solely to duplicate existing M=1 evidence**;
- mark as budget-skipped/unknown, **NOT** rejected chemistry, resource censored, failed route, nor fake success;
- do not send the incomplete path to ordinary GRPO Policy Loss (no genuine suffix return), do not give it cached Q as if newly sampled; retain for diagnostics/offline selection under a separately declared protocol;
- retain a future option to spend an explicitly budgeted exploration continuation when diversity, expected added distinct chemistry or complete-route discovery justifies it;
- original independent G4 trajectories are never pruned, even if semantic duplicates: their true future returns belong to distinct real episodes.

This is **NOT mathematically identical** to the old all-suffix search distribution. Version the new selection/stopping contract and compare at fixed actual model/token budgets; no unsupported claim of unchanged learned estimator due to dropping exploratory samples.

Avoid trying to force "different FLOW" by action masks / rejection sampling while still calling the distribution \`pi_old\`. If conditional proposals are redesigned, explicitly account for the actual sampling distribution; first version keeps original full tool distribution.

Optional next-run fork triage: prioritize parent nodes with (a) pending second distinct FLOW, (b) available unexpanded budget, (c) known nonduplicate exact state, then cap exploratory work. No hidden reference graph or future route reward can be injected into sampling decisions. If too few second distinct FLOW candidates are sampled, record as eligibility/coverage, not silently loosen conditioning.

## 4. P1 — dynamic four-GPU scheduling (this is the largest avoidable idle tail)

### P1a — minimal architecture change: G4-target-level dynamic owner queue

Replace fixed \`own=[byid[k] for k in chunk[rank::world]]\` with rank 0 emitting one ordered list of **target groups** (one source row plus its four original G4 episodes and ReFIT expansions). Ready ranks claim an unowned target group from a centralized in-memory coordinator; claim/finish records are durable with \`round_id, target_id, rank, actor_sha, contract_sha, lease_id, completion_sha\`. Each G4 group is owned by exactly one CPU state owner, reward scorer and local optimizer rank. Rank that finishes claims another target rather than waiting while another rank has a backlog.

- immutable \`policy_version\` across entire collection round; optimizer/checkpoint after ALL expected target groups are fully saved and counted once;
- preserve original task ordering/seed namespace, deterministic request_id-derived RNG independent of worker/rank, and grouping by original target identity; do not include rank in request identity;
- keep gradients globally reduced with same fixed \(N \times G \times C\) denominator even when target counts per rank vary;
- no duplicate target ownership; resume can reclaim only uncommitted leases from verified durable journals;
- identify and test how a rank with zero eligible groups still participates in collective gradient sync.

This low-risk step alone reduces average load imbalance. It will not fully fix the final pathological target tail: fast GPUs may still wait for a long group on one rank.

### P1b — solve tail too: global inference request queue

Decouple **CPU episode owner** from **GPU inference owner**.

\`\`\`text
           Immutable round policy snapshot / actor SHA
                              |
        Target Group/CPU Episode Owner + Executor threads
                    | ready prompt token request
                    v
        Global version-keyed request dispatcher
            /       |        |        \
         GPU0     GPU1     GPU2      GPU3
            \       |        |       /
             durable real token/logprob receipts
                         |
                 original CPU owner
             (executor, ReFIT, reward)
                         |
                 group post-stop scoring
                         |
                 normal DDP learner update
\`\`\`

All GPU replicas for the round run the **same frozen actor** with identical tokenizer/tool contract; a ready policy request from any CPU owner may execute on any available GPU. Use lightweight async RPC or message queues for prompts and receipts, **not per-token Ceph file polling**. CPU owner remains authoritative for chemical state, AND/OR obligations, journal, route evidence and ReFIT branching. GPU executes only policy forward/sampling and returns its original full receipt.

The dispatch policy can choose a GPU with least **estimated queued token work** and available KV cache; avoid guessing exact job cost from just target count. Scheduler should support length-aware admission/padded-token budget, GPU device memory ceiling, bounded pending queue, fairness among targets and starvation avoidance.

Important invariant: no learner rank begins optimizer update until ALL selected round request receipts are reconciled under the same behavior policy version. Preserve writer ownership, durable idempotent request IDs, recovery and one commit per group. Don't overlap **different actor versions** in the same on-policy round. A cross-device request can have different floating-point sample outputs; require valid recorded sampling logprobs and distribution parity, not assumed bit-for-bit token identity unless the counter-based RNG protocol proves it.

The model can still use 4 H20; **do not steal the other existing 4 H20** assigned to the separate D1 run. The path needs no second training model and no new reward architecture.

## 5. P2 — inference engine throughput (after orchestration)

### P2a — dynamic continuous batching / EOS refill

Current HF \`model.generate\` is synchronous per batch and cannot refill a finished row with a new request until the batch ends. It also groups by **exact remaining token room** and processes room bins in sorted order, fragmenting batches.

Prefer serving one request as:
\`\`\`text
(request_id, behavior_actor_sha, prompt_ids, allowed_output_tokens, seed)
 -> (generated_ids, per-token ORIGINAL distribution logps, eos/truncation, timings)
\`\`\`
with asynchronous \`submit/future\` and continuous refill when individual requests hit EOS, respecting independent max_context/max_action_tokens. Preserve original tool-action EOS and prompt ID parity. Guarantee each row's budget independently; no overly long request may force a shorter neighbor to become resource-censored.

Paths:
- **Choice 1**: adapt existing VllmRouteSampler after validating Qwen3.5 model architecture, FLA/BF16 compatibility, exact native prompt/tokenizer/template, temperature=1/top_p=1/top_k=0/no masks, true per-token behavior logprob capture and seeded per-request distribution. Current vLLM code exists but **is not validated/deployed**.
- **Choice 2**: implement a GPU-local continuous batching scheduler around HF forward/past_key_values with per-request KV and stop states, then benchmark against existing HF \`generate\`. Do not promise that this is easier than validated vLLM; maintain a golden HF fallback.

### P2b — avoid redundant GPU/CPU work in sampling

The current \`PrivateSampling\` computes softmax, runs a Python \`torch.multinomial(row,generator=g)\` loop across requests every token and synchronizes on \`bool(valid.all())\`; \`TokenLogprobs\` separately computes log_softmax. Investigate a single stable sampling/log-prob kernel or batching backend that:
- computes correct categorical distribution and selected-token logprob from the SAME original logits, before any token-pinning or sampling masks;
- retains per-request independent RNG keyed by immutable request ID + decode index;
- does not silently convert to greedy/top-k/top-p/temperature-changed sampling;
- exposes invalid probability detection with durable diagnostics;
- is numerically checked against the existing policy probability distribution and \`token_logps_batch\` for all generated tokens.

Do not sacrifice probability correctness merely to vectorize. Distinguish reduced Python launches, backend GPU time and real throughput.

### P2c — prefix KV

Sibling forks often share the same system/tool prefix and accepted-action-history prefix. Add version-keyed prefix KV caching only where **exact prompt token prefix** matches. Key includes model adapter/version, tokenizer/tool schema, precision, exact prefix token IDs. Full history may diverge even if chemical state is the same; no cross-history KV assumption. Cache is per actor version; invalidate after checkpoint update. Record cache hit/prefill token savings.

### P2d — durable journal I/O

Preserve write-ahead \`gpu-requested\` and immutable \`gpu-returned\` before chemistry execution. Keep per-request behavior receipts; batch fsync at verified batch/transaction barriers where crash recovery still recreates **exact** original tokens, prompt binding and probability receipts. Do not trade crash recovery or tamper detection for apparent speed. Quantify \`io.journal_flush\` and \`decode_receipts\` before optimization. Shared storage Ceph should not be used for high-frequency token-by-token RPC.

## 6. P3 — learning phase optimization

Currently \`loss_batch_size=1\`, with reference + actor forwards and backwards per compacted training entry. Next-run opt-in:
- bucketing/padding-aware microbatch sizes 2/4/8 conditioned on measured VRAM and max padded token constraints;
- compare full token-level clipped objective, fixed \`N*G*1024\` normalizer, real behavior logprobs, reference KL, compaction multiplicity and gradient synchronization against the batch=1 version (numerical and parameter-delta checks, including opposing advantage signs);
- optional frozen-reference logprob LRU keyed on exact (prompt_ids, generated_ids, reference adapter SHA, tokenizer/precision/mask) across rounds; cache exact immutable reference outputs, never actor logits used in gradients;
- preserve source receipts and each real loss contribution; batching is not reward merging.

Only optimize training after the dominant sampling cost is controlled.

## 7. Required telemetry and comparison

Per physical round AND per target AND per GPU log:
- physical model generation requests and generated tokens, prefill/decode tokens, realized active decode batch size, padded tokens, time breakdown, throughput requests/s and tokens/s;
- original G4 requests vs ReFIT credit-lane suffix requests vs exploration-lane suffix requests;
- distinct decision node IDs, semantic edges, total duplicate proposals, accepted duplicates, rejected duplicates, deterministic cache hits, in-flight shared requests, stopped extra suffixes and completed continuations actually used for M=1 Q;
- successful closure and reference-supported target counts **at same actual budget**; comparisons with extra search must retain nested-budget labels;
- conditional groups nonmasked, with nonzero Advantage, zero Advantage, fully failed and censored; measured useful unique receipt counts;
- CPU step/executor mean and tail latency, scheduler ready depth and queue latency, target/job per GPU, rank collection wall, GPU productive inference time vs collective waiting, learner time, GPU memory peaks, recovery bytes/journal count;
- checkpoint hash / behavior actor SHA / contract SHA / stock + DATA split and test identities.

Success means: (a) **less real backend work and/or less wait** for equal declared sampling/credit coverage; (b) unchanged reward conservation and same-state Q/Advantage on identical fixed receipts; (c) no fake outcomes, missing failed proposals, censored-to-failure coercions or policy-probability provenance violations; (d) route/VALID results compared on frozen data and matched actual inference budgets. **GPU-util alone is not an acceptance metric.**

## 8. Execution sequence and new-branch contract

**Phase A (lowest risk, highest priority):**
1. Add immutable read-only benchmark of existing PR110 receipts with duplicate classification: duplicate semantic proposals, expensive repeated *accepted* chemical execution, extra suffixes not used for M=1; measure actual avoidable work before setting expectations.
2. Build immutable unique proposal vs semantic edge vs continuation/credit ledger, accepted deterministic cache with single-flight, plus deterministic replay tests.
3. Separate M=1 credit lane from additional exploration, preserve G4 exact rollout; introduce honest \`duplicate_pruned_no_continuation\` state only for extra ReFIT proposals in a newly frozen sampler condition.
4. Add target-level dynamic queue with single ownership, full-round barrier and unchanged loss normalization.
5. Compare the new condition with PR110 frozen actor and a matched budget/trace scope; do not spend time on broad unrelated CI.

**Phase B (GPU utilization):**
6. Add global GPU inference dispatcher with CPU-owner/GPU-worker separation.
7. Prefer validated continuous serving (possibly vLLM) and exact per-request logprobs/EOS. If unavailable, improve HF sampling kernel/bucketing without changing distribution.
8. Batch-aware training and frozen reference cache.

**Phase C (research results):**
9. Reuse the new accelerated condition as a *new independent* D3 Standard-D training job only with explicit authorization; no in-place live updates.
10. Keep fixed B/C/D comparison, full VALID, public Reference/n5, and mandatory final TRAIN-only search-to-policy distillation, unchanged as scientific objectives.

### Focused tests (minimum)

- Same state/action with two *real* receipt IDs: one pure execution call, two individually provenance-bound proposals; no copied logprob/token or reward.
- Same chemistry but different accepted history/raw action or route obligations: no false model-suffix cache hit; context-specific FINISH guard still runs.
- 2 concurrent duplicate accepted actions: single-flight execution resolves and no deadlock; recovery from midflight crash preserves idempotence.
- Failed duplicate proposal preserves real action receipt and bounded repeat cost without fake child state or repeat Executor call.
- Extra duplicate-pruned ReFIT proposal never appears as negative chemistry nor as a completed route/GRPO sample. Original G4 unchanged.
- All sibling Q and original G4 fallback numerically match on identical completed, replayed receipts.
- Shared-prefix token policy loss once, different behavior probability/Advantage contributions never incorrectly merged.
- Dynamic target dispatch: all declared targets processed exactly once; no lease leaks on rank interruption; one actor version per round.
- Dynamic GPU request routing: unique request completion, real per-request logprobs; model output journal reconciliation and actor SHA validation.
- Variable context lengths and EOS: no neighbor-context censor, per-request stop; replay token/parity tests.
- Reordered/batch-shuffled requests maintain valid policy distributions and full action-token probability receipts.
- Learner microbatch 1 vs larger reproduces total loss and gradients to declared numerical tolerance and preserves KL/ref masking.
- Censored branches do not become failures or contaminate sibling groups.

## 9. Non-goals

- Do not change FlowER Stage-II actor initialization, reference, dataset, Reward/alpha=0.5, Process GRPO/ReFIT-PO formulas or FINISH semantics.
- Do not introduce RetroChimera, forward FlowER, DPO, Critic, PRM, new objective or unrelated paper rewriting.
- Do not claim raw duplicate-edge record merging equals suffix recomputation savings.
- Do not turn cached random suffixes into independent Monte Carlo continuation evidence.
- Do not use PUCT deterministically selected proposals as native on-policy GRPO.
- Do not replace actual G4 independent root episodes with synthetic replicas.
- Do not change/stop/restart the 2026-11-16 frozen D3 formal job or the separate D1 4×H20 job.
- Do not assume new GPUs are free; this acceleration is designed to reuse the same 4 H20 assigned to the future condition.

**Intended scientific invariants**: exact executable state grounding; route AND/OR outcome and failure semantics; reward and credit provenance; real behavior-policy sample; branch credit replaces G4 only where eligible; TRAIN-only search-to-policy distillation remains the final required study.
