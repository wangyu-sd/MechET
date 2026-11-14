# PR105 handoff readback and G0 audit

Read complete new contract, checklist and machine gates at head83a14fb79984cc1dfc67cb351837026bfdd4176b; stacked PR104 base7ce2e69ab0ff9d4803bf9690e16e27242b8dd620 is unchanged. No formal reviewer requested code changes; the latest addition is a follow-on design/execution handoff, not implemented G1–G6 functionality.

## G0 evidence

All435 files in the live eight-H20 frozen source match snapshot ef2e01f034c59182ca329ad5827172c420c6b33180b0861afee36e69df9d4d19 / local sourcec6d3fb3. CPU encoding and cache-gated queue driver are alive. Model lineage remains old Qwen3-8B History-v2 checkpoint31366, parent SHA a0f67c8ab799e5dad18ba97d824a7d4ae2a0280083eeeaca75c48ec903028c9b. No queue restart, source mutation, new GPU run, changed reward, or new campaign was performed by this audit.

## Gaps confirmed against current source

- Exact lookup includes full history identity. Existing transferred advice checks the error family and stock/executor versions, but has no separate legal-inventory/active-obligation eligibility query or empirical conflicting-support aggregation. G1 remains necessary.
- The current memory episode still prepends the full v8 ledger in each prompt. Retain that old contract for replay; the bounded four-view observation is a new opt-in version. G2/G3 must retain every raw event outside the prompt, while mandatory current state/route obligations remain present.
- The current five-advice-item limit is not an actual Qwen-tokenizer budget. Proposed512 optional tokens/3advice/5failures are profiling defaults, not established improvements.
- Frozen per-episode memory already exists. New group-wide actor/memory/retrieval-policy bindings and per-decision rendered-memory metadata still need explicit verification.
- LangGraph and verl are not installed in the current host meteor environment. No framework adapter/parity is claimed. Install optional, pinned dependencies in an independent Ceph environment when implementing the adapter; do not alter the running cache/queue runtime.
- A LangGraph adapter should pass native tool requests/receipts through unchanged, use per-sibling durable references and leave chemistry/termination to RouteEpisode. A verl bridge must preserve original per-turn token IDs/logprobs; re-encoding final chat messages is insufficient, especially when observations rebuild current state each turn.

Official interfaces inspected: [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [LangGraph memory](https://docs.langchain.com/oss/python/langgraph/add-memory), [verl AgentLoop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html). No versions have been installed/pinned by this readback, and no runtime API compatibility or GPU gain is asserted.

G0 source/status audit is complete. G1–G4 remain follow-on implementation; G5 is optional after parity requirements and G6 requires actual corrected8B artifacts/full declared validation. The original training sequence and8H20 ceiling remain unchanged.
