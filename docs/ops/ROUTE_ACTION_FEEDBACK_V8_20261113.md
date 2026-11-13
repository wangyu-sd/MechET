# Route action feedback v8: implementation and acceptance

The long-route diagnostic exposed stale historical atom aliases, premature FINISH and repeated rejected execution. The original checkpoint's ability to execute local electron-flow programs did not establish whole-route convergence. V8 repairs the decision interface while preserving the original chemistry, stock, reward and complete evidence archive.

**Status: the declared limited paired H20 gate passed. Source-bond errors fell13→0, commits3→8 and inventory closure0/4→2/4.21 repeated model proposals and0 reference closures remain; the planner is not fully repaired. Formal v3 training is unchanged.**

## Findings

All196 saved proposals replayed exactly.64 rejections did not mutate the valid node, and15 new-target transitions produced exact fresh roots.18 absent-bond errors were independently confirmed in the current RDKit graph. Thus this audit did not find stale terminal flags or a mismatch between the displayed current inventory and the executor.

In full-history v3,13of14 absent-bond failures referenced aliases that had formed bonds in prior states and had appeared in accepted electron flows. Axx aliases are regenerated at each decision. Historical overlap is evidence of a scope risk, not proof of attention/copying causality.45 FINISH proposals accepted by the bottom executor were rejected by route guards:16 ancestor cycles and29 nonproductive endpoints. FINISH acceptance alone is not chemical correctness.

The previous long smoke used two fixed TRAIN roots with two replicas per condition: native control131 decisions/13 commits/1 inventory closure, full v3 65/3/0. Both had0 reference-supported closures. A reference-unsupported alternative remains unverified, not proven chemically wrong. These are diagnostics, not benchmark or generalization results.

## Repair

`route_action_feedback_v8` is opt-in. The original event archive retains every state, observation, action, output and failure. Its model view uses the v7 opaque per-observation historical handles and structured past actions, with native current instructions last. A current execution-constraint block precedes the native current observation: finish eligibility/reason, current existing source-bond candidates and rejected semantic actions for this exact mapped state and reaction.

Each request carries its actual tool schema. Both HF and vLLM render that schema. The existing productive/acyclic guard decides whether finish_trace is advertised; valid FINISH remains available. This is a prompt/schema condition, not a logits mask or forced tool output. A model may still hallucinate unavailable tools; such proposals are retained and rejected normally. Actual prompt IDs and original categorical logprobs remain authoritative for learning. Live vLLM engine probability parity is not claimed by the CPU plumbing checks.

Repeated equivalent rejected proposals at the same exact mapped state/current reaction reuse the prior rejection, preserving the new proposal and feedback but avoiding another chemistry execution. Explanatory instructions, unordered bond/radical endpoints and executor-default empty arrays do not distinguish equivalent actions. Electron-flow sequence order and chemical parameters remain intact. State changes and other precursor occurrences do not share these failures. Failure memory is derived from receipts, so snapshot restore has no second mutable cache to reconcile.

This does not introduce MCTS/backtracking, a value model, process rewards or a new optimizer. Reusing rejected execution alone cannot establish GPU speedup, reduce model proposal counts or prove convergence. Those remain measured outcomes.

## Software evidence

18 focused tests pass. Expanded route/external checks:565 pass;3 harness failures were CUDA_VISIBLE_DEVICES=-1 being treated as a two-rank list and missing source PYTHONPATH in a child interpreter. All3 targeted reruns pass with the correct test environment. Counts overlap; they are not summed. The real-package full-project run did not finish within240 seconds, so no whole-project green claim is made.

An independent review found equivalent-action cache misses. Three real-executor regressions reproduced them RED; normalization repairs passed GREEN. Four saved failure-state interfaces preserve the exact current state and full archive, advertise FINISH according to the guard, skip repeat execution and restore exact requests. One of those states correctly keeps FINISH available; v8 does not disable it indiscriminately. These are software replays, not fresh model inference.

## Reproducible integration and GPU gate

[Integration patch](../../patches/route_action_feedback_v8_20261113.patch) applies to frozen21 identified by snapshot SHA22e2a6371e4ba2318b2d63de36dc6d575999e97f01d3614d1e27024f61b5194b. [Bindings](../../configs/ops/route_action_feedback_v8_integration_20261113.json) and [patch verification](../../configs/ops/route_action_feedback_v8_patch_verification_20261113.json) record dry-run/application and all10 resulting file hashes. Complete runtime dependencies remain in the named frozen trainer; public main does not contain that entire deployment.

Fresh v3/v8 H20 task: meteor_mechet_route_action_smoke_1h20_zjk_20261113_01, instance8b1d89f6a1cfaa4801a1d40d41540794. Same two roots, two replicas, original corrected History89442 adapter2c98a9bff3afeaffbf2ee2733885cccc68e936f9310fd9f46ba4ee2211e5f453, unchanged request-RNG namespace, batch<=4, max64 diagnostic decisions, no saved prefix or optimizer. Frozen407 sources have only10 declared deltas; snapshot48664c5192ec2b38af711587a4c281231bb847f35026937404aedf3f716b1138.

Require actual before/after GPU tensor equality/186 active modules, physical B4 and unchanged0.25 sampled probability audit gate. Behavioral evidence requires fewer repeated rejected semantic proposals, at least fresh-v3 committed reactions and stock closures, zero missing-atom errors and zero resource-censored episodes. Earlier exit alone cannot pass. Report structural closure and reference closure separately, preserve all failed gates, and do not promote formal training solely on software checks.

Previous smoke05 also ended FAIL: native7/8 commits, passive v7 4/8, max probability drift0.3581516743. Its failed report is retained alongside the long smoke; v8 does not rewrite that history.

## Completed H20 outcome

[Aggregate report](../../configs/ops/route_action_feedback_smoke01_results_20261113.json) and [actual prompt/tool/receipt audit](../../configs/ops/route_action_feedback_smoke01_audit_20261113.json). Both fresh arms used the same original actor, roots and request RNG.

| Metric | Fresh v3 | Repaired v8 |
|---|---:|---:|
| Policy requests |65|85|
| Absent source-bond errors |13|0|
| Reaction commits |3|8|
| Inventory-closed trajectories |0/4|2/4|
| Reference-supported closures |0/4|0/4|
| Repeated rejected proposals |23|21|
| Reused rejected step executions |0|21|

Both inventory-closed trajectories concern the same one of two fixed TRAIN roots (unique-root closure0/2→1/2). They remain reference-unsupported alternatives, not established chemical successes. There were no64-decision or physical-capacity censored episodes. Two v8 trajectories still ended after8 consecutive rejections.15 FINISH proposals were not in the actual advertised schema; the model can violate a prompt-level tool list.21 repeated proposals are real additional GPU work even though step execution is reused, so this result does not establish reduced GPU calls or total acceleration.

150 recorded proposals replay exactly; all150 actual prompt token sequences were reconstructed from messages plus their actual tool schemas and matched the GPU receipts. Complete prior history and commits were checked. Both arms ran physicalB4. Before/after372 FP32 source-equal adapter tensors and186 active modules passed;31 sampled cached/full probability checks max0.2306985855<0.25.0optimizer updates. Driver1054.17s.

The predeclared repeat/commit/stock gate passes narrowly (21<23,8>=3,2>=0), not a claim that recurrence is eliminated. Reference reward remains0 for all these episodes. No live vLLM engine GPU parity, process-RL benefit, full benchmark gain or formal-job migration is claimed.
