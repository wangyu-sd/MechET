# PR105 Codex execution contract: standard Agent runtime and chemically grounded memory

**Design status: proposed implementation contract, NOT a completed feature or a completed GPU experiment.**  
**Date: 2026-11-13. Repository: wangyu-sd/MechET. Branch: `train/qwen3-8b-rsi-8h20-20261113` (stacked on PR104).**  
**Companion:** [PR105 current eight-H20 execution status](QWEN3_8B_RSI_8H20_20261113.md); [PR104 v8 route-action feedback](https://github.com/wangyu-sd/MechET/blob/ops/route-rl-efficiency-codex-20261111/docs/ops/ROUTE_ACTION_FEEDBACK_V8_20261113.md).

## 0. Required Codex interpretation: preserve work already shipped

This is a **focused follow-on implementation** to the existing approved eight-H20 Qwen3-8B pipeline, not a restart or another model proposal.

Source verified in PR105 at the time of this contract:
- Corrective SFT is gated on complete 4-shard CPU cache, then 8×H20 BF16/NF4, B1×GA8/global64, old Qwen3-8B History-v2 checkpoint-31366. TRAIN 243,753 reactions / 1,908,035 decisions; VALID 2,751 / 21,355. It has **not** yet produced a reported finished new 8B GPU checkpoint. Do not disturb the waiting cache/queue and do not change the model lineage.
- `route_experience_memory.py` already provides a strict evidence-bound exact-context store with hash-bound completed TRAIN verifier receipts, deduplicated feedback, snapshot/restore.
- `route_memory_integration.py` already provides opt-in `MemoryTargetOnlyRouteEpisode`/`MemoryValidationRouteEpisode` with actual episode memory SHA binding and three **typed**, non-executable pieces of advice.
- `route_memory_curation.py` already provides TRAIN-only replay/curation and refuses to label mere accepted imports as successful recovery.
- PR104 v8 has current source-bond grounding, FINISH eligibility, native observation precedence and same-exact-state cached rejection **execution**, not action-proposal suppression.
- `RouteEventHistory` and original per-token behavior receipts remain authoritative for exact replay.
- Currently **no LangGraph/verl production adapter, no route-level memory retriever, no chemically transferable experience index, no published full-run benefit**. Do not mark them implemented merely because the existing typed advice API works.

### Non-negotiable scientific constraints
No rewrite of IMPORT/FLOW/FINISH or underlying electron executor, stock/cycle/AND obligation rules, protected splits, reference scorer, baseline G4 Dr.GRPO or ReFIT search semantics. No value critic or model-generated unverifiable reaction “facts”; no new reward terms. Never equate structurally stock-closed with experimental feasibility or reference-unsupported with invalid chemistry. Do not replace PR103 conditional forks with selected search winners. Do not change the 8-H20 overall GPU ceiling or launch a second competing 8B campaign.

## 1. Memory responsibility and safety model

**Four model-facing memory views built on existing immutable Event Ledger:**

| View | Authority / scope | What may enter model context |
| --- | --- | --- |
| Working Memory | Current executor state, current atom/bond handle inventory, active occurrence, last action result | Mandatory, regenerated from CURRENT executor. Never restored as authority from remembered text |
| Failure Memory | Current exact mapped chemical state and reaction occurrence; semantic action signatures and verified rejection reason | Small number of current-state failures. No raw obsolete handles from another occurrence |
| Route/Proof Memory | Actual pending precursor **occurrences**, accepted reaction AND hyperedges, stock leaves, ancestor/cycle guards and independently generated certificates | Compact deterministic subgraph/obligation view; no hidden TRAIN reference route or reference progress |
| Episodic Experience | Immutable TRAIN-only verifier receipts, typed recovery advice, transferable motifs with evidence and applicability | Optional, source-labelled, non-executable advice. Cannot become a new chemical constraint or hidden result |

**Persistent evidence** (all events, raw tool output, executor snapshots, original token IDs/logprobs, reference scoring receipts) is a separate durable backing store. It is *not* the prompt. Do not copy the complete ledger into each prompt; the previous v3/v8 lossless serialization is preserved for replay and backward compatibility. The newly rendered observation is a **new versioned contract**, not a byte-preserving optimization.

Memory lifecycle:
1. Exact executor state is authoritative at every decision.
2. Build Working + Route/Proof state deterministically from private executor and public stock (never from an LLM summary).
3. Derive Failure Memory from current-episode receipts. All attempts remain in Event Ledger, even if prompt shows only relevant errors.
4. Retrieve optional TRAIN experiences from a **frozen** memory snapshot.
5. Render only selected *public* memory facts with verifiable source labels; retain current chemistry observation/tool instructions **last**.
6. Log `memory_version`, `memory_snapshot_sha`, `retrieval_ids`, `rendered_memory_digest`, `prompt_token_ids`, `generated_token_ids` and true frozen-behavior token logprobs for every decision. Event/raw archive always retained separately.

## 2. Two levels of memory identity: **do not collapse them**

### Level A — Exact execution/rejection memory (machine-grounded)
Reuse the existing strict `scope_key` when exact repeat semantics require a complete execution+history context. Execution-result caching for rejected actions MUST use current mapped state, reaction/occurrence, action semantic signature, imported participants, executor/stock version and any necessary legality guards. Invalidate on state/reaction change and never reuse cached token logprobs or policy decisions.

Existing PR105 `scope_key` includes `event_history_sha256` and can be highly specific. This is safe as an exact identity but **cannot** serve as a general transferable retrieval key: two equal local chemical states with different histories will usually hash differently.

### Level B — Transferable **typed** strategy/chemical-context retrieval (non-executable)
Keep the three existing typed `strategy_id` values as a safe starting vocabulary:
- `inspect_current_source_bonds`
- `continue_before_finish`
- `avoid_route_ancestors`

Implement a **separate**, versioned retrieval query and eligibility check. Start with relevant error family + current legal-inventory facts + active reaction status + outstanding route obligation summary + source executor/stock compatibility. This is a **proposal for matching evidence**, not a guarantee of successful chemistry. A generic typed advice sentence may transfer; **no specific historical atom handles, precursor lists, full route solutions, reference labels or tool arguments** may transfer.

Optional local chemical substructure descriptors (atom-map invariant, reaction-center/bond-local) may be added **only** with an explicit matched-chemistry test and source provenance. Never use text embedding similarity alone as proof of valid actions; do not invent a reaction/template by retrieving a vaguely similar substrate. When applicability is uncertain, emit no advice.

Retrieval must be queryable outside the complete exact `event_history_sha256` key and still support complete exact lookup separately. Track source evidence IDs, split, executor/version, target identity leakage constraints, whether evidence is executor-only or reference-supported, and conflicting outcomes. For a given candidate, compute empirical support counts and expose unknown uncertainty rather than assigning a fabricated calibrated success probability. A `successes / all receipts` number is not a chemical feasibility score.

## 3. Bounded deterministic memory projection, not unlimited context

Specify a model-visible schema `route_memory_observation_v1` with separate keys:

```json
{
  "schema": "route_memory_observation_v1",
  "working": {"current_state_id": "...", "active_occurrence": "...", "legal_inventory_source": "executor"},
  "route": {"pending_occurrence_count": 0, "pending_occurrence_refs": [], "stock_closed_occurrence_count": 0},
  "failures": [{"state_ref": "...", "action_signature": "...", "error_family": "...", "count": 1}],
  "advice": [{"strategy_id": "inspect_current_source_bonds", "evidence_ids": [], "applicability": "generic_error_recovery_advice"}],
  "memory_provenance": {"snapshot_sha256": "...", "retrieval_policy_version": "..."}
}
```

This is a proposed schema, not an existing code output. Reactant structures/stock identities in tests must be source-real and permission-scoped. The entire unbounded reaction hypergraph MUST NOT be rendered; provide references and a small deterministic projection of the active obligations while retaining the exact backing graph. Order memories deterministically and sanitize text/chat delimiters as PR104 does.

Use the **actual deployed Qwen3 tokenizer** to compute memory input tokens. Proposed initial values **for profiling, not validated optima:** `max_optional_memory_tokens=512`, `max_advice_items=3` and `max_current_state_failures=5`. Record actual token counts including tool schema/templating; apply cap to **optional memory**, not to the current authoritative state and mandatory route obligations. If the required current state alone cannot fit in the physical context: record `RESOURCE_CENSORED`, not a false chemical failure, and exclude the group according to the declared G4 contract. Do not silently truncate route obligations, electron programs, original sampled completions or required current handles.

No independently sampled “memory manager” LLM in v1. The retrieval policy must be deterministic, versioned and reconstructible from evidence; a learned memory controller can only become a separate, explicit later experiment.

## 4. Standard Agent framework: official LangGraph with optional verl bridge

**LangGraph:** implement a small `StateGraph` adapter, not a second independent chemistry simulator. Current `RouteEpisode`/electron executor owns ALL state transitions and stop predicates. LangGraph owns scheduling boundaries, checkpoint refs, and node transitions:
`build_observation → sample_native_tool → execute_existing_tool → ledger_and_memory_update → next_occurrence_or_stop`.
It must not fabricate a separate model plan, auto-select alternative tool calls, or modify model outputs. Hidden reference scoring is called **only after observed terminal**, outside the model-visible graph. Use explicit typed graph state with `task_id, episode_id, group_id, sibling_id, behavior_version, memory_sha, executor_snapshot_ref, pending_occurrence_ref, request_id, tool receipt refs, terminal/censor status`. Keep big molecule/route artifacts in existing private, durable shared storage; state/checkpointer stores references, not expanded repeated prompts.

Use a supported pinned official LangGraph release and official durable checkpointer abstraction. Unit tests may use in-memory saver; production must use a durable supported backend (or a **tested adapter** to existing atomic snapshot store). Treat `thread_id` as episode/sibling-scoped to prevent cross-target and cross-rollout writes. Do NOT make cross-run LangGraph Store automatically readable to the four independent training samples.

**verl:** official `AgentLoop` is an OPTIONAL, second-stage rollout/learner backend, not a required rewrite before the current corrected-8B SFT. Its docs note the AgentLoop interface is evolving. First implement a thin `AgentLoopBase`-style adapter preserving generated token IDs, assistant-only mask (tool/prompt tokens excluded), actual behavior logprobs, original multi-tool prompt segments, terminal/censor semantics, frozen G4 advantages, KL and distributed normalization. Compare numeric loss and parameter gradients against existing Dr.GRPO with the same frozen trajectories; reject mismatches. Do not silently substitute standard GRPO for route-Dr.GRPO. Benchmark GPU utilization/overlap before switching. Keep existing backend as rollback baseline.

Official design references (read and pin actual installed versions before coding):
- https://docs.langchain.com/oss/python/langgraph/persistence
- https://docs.langchain.com/oss/python/langgraph/add-memory
- https://verl.readthedocs.io/en/latest/start/agentic_rl.html
- https://verl.readthedocs.io/en/latest/advance/agent_loop.html

**Compatibility/packaging:** official-framework imports must be optional extras; existing `main`, 0.8B history/route runtime and 8B CPU token cache must remain importable and unchanged without LangGraph/verl installed. Avoid adding background GPU services or changing active queue source snapshots. Declare exact lock/commit/license and dependency versions; verify local runtime compatibility and policy-token provenance before live integration.

## 5. ReFIT/UDS-style route graph connection (no search-policy rewrite)

Reuse `RouteNetwork` AND/OR graph and exact `RouteEventHistory` receipts; do not build a second full route planner as “memory.” Provide a read-only `RouteProofMemoryView` giving current structural obligations, source accepted reaction edges, stock status and available certificates, with occurrence multiplicity preserved. This view is shared as an **interface** between training Episode state and ReFIT inference but **not a shared mutable store across G4 siblings**. 

Think of the ASKCOS graph/pathway separation and Syntheseus's search/evaluator interfaces as engineering precedents (do not import their full dependency trees):
- full executed electronic programs -> reaction hyperedges -> route certificates
- model-facing `RouteProofMemoryView` is only a small public projection
- stock closure is mechanical, not independently chemically verified
- original search frontier/reaction backup remains unchanged until a separate properly budget-matched ReFIT ablation
- hidden TRAIN reference routes, scored edges, validation/test labels never enter the agent prompt or global memory

## 6. RL/statistical boundary: this is not free RSI

Four root trajectory samples per target must be independently generated under the same **frozen behavior actor** and an explicitly frozen memory snapshot/version for the whole G4 group. Per-episode private Working/Failure/Route memory is writable only inside that episode; any TRAIN cross-episode store is read-only during collection, not updated after sibling 1 so sibling 2 can read it. Resume must fail if actor, memory, stock, observation version or prompt token/hash binding differ.

Persist `(prompt_ids, sampled_completion_ids, behavior_logprobs, memory_sha, retrieval_policy_version, generated_action, executor_receipt)` for every tool call. If memory or LangGraph observation changes, old behavior logprobs are **not** probabilities of the new prompt. Fresh collection is required for any on-policy comparison.

Avoid leaking hidden reward progress through advice. Verified historical TRAIN-only examples may be read under a frozen split-safe retrieval policy, but protected targets/identities remain excluded and memory is not updated based on VALID/TEST outcomes. FINISH/source-bond prompt guidance remains **advisory** unless the training/inference action distribution and logprob estimator are explicitly changed. Resource-censored siblings remain unknown, not chemical failures. MCTS winners are offline data, not root on-policy samples; PR103 conditional forks are a separate estimator and are not silently enabled by this refactor.

## 7. Codex implementation phases and mandatory acceptance tests

| Gate | Files / deliverables (suggested) | Acceptance |
| --- | --- | --- |
| **G0 Source & status** | Verify latest PR105/PR104 head, named private deployment, current 8H20 queue state; snapshot SHA/versions | No fabricated done/running claims, no modification of live 8B queue; preserve stack base |
| **G1 Memory records** | Extend `src/mechet/route_experience_memory.py` with explicit exact and typed-transfer APIs; add `src/mechet/route_memory_views.py` or similarly scoped adapter; unit tests | Exact hash reuse unchanged; generic advice has provenance/eligibility, excludes stale handles/hidden labels; unknown returns empty |
| **G2 Route/Failure projection** | Integrate immutable ledger, exact current-state failure index and public route obligation view into `route_memory_integration.py`; test replay+restore | Every old event preserved; current executor status authoritative; duplicate AND occurrences retained; no cross-reaction/stale handle reuse |
| **G3 Deterministic retrieval** | `route_memory_retrieval.py` + versioned observation/config; actual tokenizer tests | Initial optional memory cap and evidence-aware deterministic ranking; no silent truncation; v8/no-memory parity when flag off |
| **G4 LangGraph adapter** | `agent_runtime/graph.py`, `state.py`, `checkpointer.py`, adapters/tests; no separate chemistry model | Same input tools/observations and accepted/rejected transitions as native control; exact checkpoint/resume; independent sibling threads |
| **G5 RL backend parity** | Optional `verl` adapter and tests, only after G1–G4; keep original Dr.GRPO fallback | Full prompt/token/behavior-logprob/masks, conditional off, freeze G4, KL/clip/all-reduce and terminal/censor matching; synthetic + genuine saved-trajectory numeric and gradient diff tests |
| **G6 Frozen comparisons** | Baseline summary, replay/evaluator reports, full-versioned new experiment condition | No performance claims from CPU replay or 2-target smoke alone. Full declared VALID and matched GPU budgets required for route quality; 0.8B and 8B populations cannot be conflated |

**Minimum required regressions:**
1. Same mapped state + same reaction + same semantic invalid action -> deterministic rejection cache hit; change the state, imported atoms, ancestors, executor version, or occurrence -> no unsafe hit.
2. Same chemical subgraph but different historical atom maps -> transferable *typed advice only*, never a stored executable FLOW.
3. Full exact history SHA mismatch -> exact lookup miss while eligible generic advice may still appear; without independent evidence -> no advice.
4. Long route with several FINISH boundaries, repeated AND precursor occurrences and 8 previous rejected actions -> faithful replay/restore and correct current source-bond inventory; no old-state instructions at end.
5. Prompt construction with complete archived history of arbitrary length -> bounded optional memory under **actual tokenizer**, full archive recoverable, mandatory current state not silently truncated.
6. G4 group snapshots identical actor/memory version; no within-group knowledge leaks; VALID/TEST cannot write; TRAIN-hidden reference never reaches prompt.
7. Programmatically tampered evidence file, forged memory ID, stale receipt, or wrong source snapshot -> fail closed, including after restore.
8. Old v8 no-memory and corrected 8B runs remain runnable with no optional framework dependency.
9. Under the same frozen actor and inputs, LangGraph adapter matches native executable transitions/termination and retains original generated token IDs.
10. If verl enabled, route Dr.GRPO numeric and gradient semantics match current implementation before any GPU training.

**Primary engineering/learning metrics:** tokenizer-counted prompt and memory tokens; repeated rejected **model proposals** vs cached executor invocations (distinct), failure recovery, invalid FLOW, unoffered FINISH, memory retrieval hit/precision with evidence, unique source scopes, G4 whole-group censor rate, nonzero centered-advantage groups, completed reaction occurrences, structural vs reference-supported stock closure, actual calls/tokens/GPU-hours and full VALID solved rate. All denominators explicit. Compare at the **same corrected Qwen3-8B actor**, same memory version within an arm, same target set and train/validation protocol.

## 8. Delivery contract and stop conditions

- **Code** must be committed to **this PR105** on `train/qwen3-8b-rsi-8h20-20261113`; update docs/results with current status and exact commit/experiment identities. Do not open a fresh PR or silently merge PR104.
- **First deliverable**: exact vs transferable memory design, tests, route/failure projection and read-only offline replay under complete existing snapshots.
- **Second deliverable**: LangGraph optional StateGraph adapter and checkpoint/resume parity under native same-action tool receipts.
- **Third deliverable**: versioned memory-conditioned observation integration, tokenizer budget, independent G4 audit; stage-limited official RL backend adapter only after tests.
- **Fourth deliverable**: matched performance evidence with frozen weights; do not claim scientific improvement until full official VALID and route metrics exist.
- Keep active corrected 8B cache/queue work independent and uninterrupted. It is acceptable to declare G5/G6 blocked awaiting actual new weight SHA and valid evidence rather than inventing trials.
- If no official framework available in active env, prepare reproducible installation/lock instructions and a CPU replay adapter; **do not fake installed dependencies, GPU tests or rollout parity**.
- Stop on missing verified TRAIN receipts, unverifiable chemistry source, new hidden-label leakage, sampler/logprob mismatch or catastrophic memory/context overflow. Preserve failed results and rollback path. Never convert a resource limit to an intrinsic chemical negative.
