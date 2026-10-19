# Conventional complete-reaction planner: existing, not currently matched

Read-only inspection found real existing code, not a missing implementation:
`scripts/run_syntheseus_online.py`, `scripts/run_syntheseus_search.py`,
`src/mechet/online_syntheseus_adapter.py`, `src/mechet/syntheseus_adapter.py`
and the separate offline proof-pool `scripts/search_proof_routes.py`.
The [actual configuration/availability audit](availability_audit.json) records
the instantiated legacy parser/policy arguments and all four source hashes.
No model or planner ran. Optional Syntheseus is absent in this local Python
environment; that observation says nothing about available GPU resources.

The online entrypoint selects the same0.6B base revision and can request
compact history, but it is not a current frozen-condition control:

| Boundary | Existing legacy entrypoint | Archived actor / new prepared independent control |
|---|---|---|
| Model context |4,096 by default|8,192 |
| Electronic decision/import counts |40 decisions/32 imports; enforced in adapter constructor|No corresponding chemical decision/import-count cutoff |
| Decoding backend |Older Transformers runtime; `no_4bit=False` by default|Pinned vLLM0.8.5 V0/TP1 BF16 actor runtime |
| Outer default work |100 reaction-model calls/1,000 iterations/300s|Original actual work separately audited; no matched-work equivalence established |

The legacy reaction conversion also splits every dot component into a separate
Syntheseus molecule. That is not the present route controller's special handling
of an intact whole-salt stock entry. The offline proof-pool entrypoint expects
complete `MECH_PROOF` strings, whereas current saved actor programs are native
natural-language tool trajectories. They cannot be relabelled by renaming files.

Thus the existing code is preserved, but cannot be called a matched completed
R1-C experiment. It would require a current-native complete-reaction provider,
stock semantics and authenticated full-population/work contracts before a
separately authorized model run. A retrospective search over MCTS's already
discovered full reaction pool would be a different offline reachability
diagnostic, not an independent online conventional-planning baseline.

This is the explicit unavailable-control state permitted by LGRV section7 and
R1-B's “when runnable” condition. It does not cancel the planned comparison or
turn its missing output into failure. Available same-target Greedy/MCTS
literature comparisons remain intact and recorded-work descriptive only.
There is no new empirical baseline, planner speed or chemistry claim here.
