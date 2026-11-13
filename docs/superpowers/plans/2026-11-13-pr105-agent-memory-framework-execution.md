# Codex plan — PR105 chemically grounded Agent Memory + standard AgentLoop

**Status:** this checklist is a plan, not evidence of implementation. **Primary contract:** [PR105 Agent Memory architecture](../../ops/PR105_AGENT_MEMORY_FRAMEWORK_CONTRACT_20261113.md).

**Preserve:** existing PR105 Qwen3-8B, full corrected data, cache/8H20 queued job, completed exact-evidence memory API, typed-advice memory integration and real TRAIN curation. Existing PR104 v8 history/chemical-state fixes remain the base; do not restart model/queue or alter reward.

## Deliverables (execute in this PR)

- [x] **G0 — Rebase safety and provenance:** inspect newest PR105 head and PR104 base, frozen deployed source snapshot and current queue. Write status/versions/readiness; do not mutate queued active training source.
- [ ] **G1 — Two-tier memory:** keep `ExperienceMemory` exact execution key; add separate typed transferable advice retrieval with explicit error/current-inventory/active-obligation eligibility and evidence IDs, deterministic ranking, refusal of obsolete atom handles. Regression: different histories same local state -> no unsafe executable reuse; typed advice may transfer.
- [ ] **G2 — Four public state views:** deterministic Working/Failure/Route-Proof/Episodic memory projections backed by exact v8 Event Ledger. Preserve all original events, repeated precursor occurrences and route certificates; never expose hidden reference route. Typed advice is optional, not chemical truth.
- [ ] **G3 — Context budget and binding:** version `route_memory_observation_v1`; render true Qwen3 tool/prompt template; cap optional memories with tokenizer tokens, never silently drop current state/required obligations. Freeze memory snapshot/retrieval policy for each G4 group. Persist actual prompt IDs, per-token behavior logprobs, memory and retrieval hashes. Restore must fail on version mismatch.
- [ ] **G4 — Official LangGraph StateGraph adapter:** runtime owns nodes/checkpoints, but existing RouteEpisode and native executor remain transition/stop authority. Write durable checkpointer isolation; prove same-action deterministic parity, complete tool receipts and independent sibling threads; package framework dependency as optional.
- [ ] **G5 — Optional verl AgentLoop:** after G4 and with explicit profiling need, implement token-ID-preserving rollout bridge; test route Dr.GRPO PG, KL, mask, censor, global normalization and gradient equivalence. No silent migration to standard GRPO or full-async off-policy.
- [ ] **G6 — Matched evaluation:** old PR105 exact+three advice, new retrieval/memory, and memory-off conditions all use same corrected8B checkpoint and declared target/split/budget. Audit failure recovery, repeated model requests, context censor, valid group yield, route stock/reference closure, GPU work and full VALID; do not infer gains from isolated successful roots.

## Required Codex report

For each gate commit exact code, software test command/output, source/actor/tokenizer/memory/checkpointer SHA, counterexample replays, input prompt/token counts, closed/open blockers and any changed protocol namespace. Advance only upon passing gates. The stack stays Draft. Do not claim official runtime integration, GPU success or 8B model improvement merely because software tests pass.

Official interfaces for implementation research: LangGraph StateGraph/persistence docs `https://docs.langchain.com/oss/python/langgraph/persistence`, verl AgentLoop `https://verl.readthedocs.io/en/latest/advance/agent_loop.html`. Pin actual compatible versions rather than assuming API stability.

G0 readback: all435 deployed source files matched the existing snapshot; live CPU/cache-gated driver preserved. Evidence `configs/ops/pr105_g0_readback_20261113.json`, interpretation `docs/ops/PR105_HANDOFF_READBACK_20261113.md`. G1–G6 remain unimplemented in this handoff update; no new runtime/chemical gain claimed.
