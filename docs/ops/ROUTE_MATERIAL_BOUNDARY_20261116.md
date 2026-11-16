# Mechanistic endpoints and material availability

The FlowER example `flower_mech_proof_train_37056` correctly ends at methylthiolate plus an alkyl bromide. The original forward reaction already starts from the thiolate; its conversion to SFT did not omit a proton-transfer step. The route adapter nevertheless reopened `C[S-]` as a synthesis obligation because the frozen stock contained `CS` rather than that ionic fragment. A reagent can contribute atoms to the product and still be a purchasable starting material: contributor selection and availability are separate decisions.

Sodium methanethiolate is an actual commercial reagent ([supplier](https://www.sigmaaldrich.com/US/en/product/aldrich/281018)). This supports the chemical distinction between an active reagent and an unfinished connected intermediate; it does not establish general stability or availability for every charged prediction.

## Declared repair

The unchanged endpoint contract `target_heavy_atom_contributors_v1` still extracts all fragments containing target heavy atoms. A separate opt-in `stock_material_contract=terminal_heteroanion_protonation_stock_v1` projects a narrowly defined conjugate base onto a stock material. Missing/omitted configuration retains `exact_material_stock_v1` and historical behavior.

An alias requires a single nonradical fragment, exactly one charged atom with charge -1, that atom being terminal O or S with no H, bonded singly to carbon. Add one ordinary H to that atom, leaving every heavy-atom bond, stereo label and isotope unchanged, and require the exact resulting structure to be present in the frozen stock. Exact original-stock matches take priority. There is no general uncharging, tautomerization, redox correction, counterion deletion, or alias for C-/N-/cations/radicals/multicharged fragments.

This is an explicit protonation/workup **availability assumption**, not an executed electron move. `final_state`, mapped atom identities, and predicted actions retain the actual charged mechanism. Each committed edge records `endpoint_materials` with the mechanistic identity, material identity and rule. Independent occurrence multiplicity survives even when alias products could be joined into a stock mixture. Exact whole-stock mixtures retain their original priority.

The same resolver supplies commit children, projected cycle checks, stock stopping, hidden reference reward, full VALID scoring and route certificates. Scoring reconstructs endpoint evidence from full initial/final states and checks per-map element/isotope preservation. Opt-in snapshot restore replays all original requests in every supported observation mode. A mismatched material contract or forged endpoint record is an error.

## Result and limitations

Four saved TRAIN boundary witnesses were replayed using their original actions, observations, acceptance/error and mapped state evidence. In the thiolate witness, the route now closes after its first 8 saved decisions rather than executing the extra 3 decisions used to process the already available active reagent. Its electron program and charged final state remain identical. Certificate reconstruction and full action replay pass. Three connected charged intermediates receive no aliases and remain pending.

The thiolate witness's full route remains `unsupported_closed`, reward 0: an earlier upstream edge is absent from the reference graph. The mapped thiolate/material edge itself matches a reference edge, but trusted ancestry cannot cross that unsupported parent. This is correct separation of stock closure from reference-supported route success; no experimental success-rate improvement is claimed.

The live 20261115 runtime, stock, reward, task and checkpoint contract have not been edited. Future use requires a newly frozen condition carrying the material contract. This change does not train failure recovery, prevent neural resampling of rejected actions, establish chemical maturity for arbitrary intermediates, or validate reference-absent alternatives. Those remain separate work.

## Actual rollout failure causes

The first 28 saved TRAIN rounds contain 4,792 unclosed episodes. Their terminal causes are route ancestry cycles 2,222; no-net-change FINISH 886; within-reaction state cycles 688; missing destination/source bonds 704; invalid chemistry 274; other 18. Of the unclosed episodes, 3,904 have eight identical consecutive rejected actions and 3,096 terminate with eight rejected FINISH calls. These are actual execution/recovery failures, not explained by absence of a reference answer.

The v9 recovery projection includes only the last failed call and its error. At unchanged state, later retries can therefore have identical prompt token IDs; saved request IDs/seeds show these are distinct model calls, not an execution cache returning the same result. SFT did not teach failure recovery. This patch fixes the demonstrated reagent handoff; cumulative failure evidence and recovery supervision remain the next implementation target. Another concrete parser mismatch is a nonexistent destination described as an existing bond rather than a bond to form; nonexistent electron-source bonds require a different valid action and cannot be normalized into existence.

Separately, a route can close in stock through a different disconnection and receive no reference-supported success credit. That concerns scoring of already closed routes, not why an executor rejects an action. The counts above precede this patch; no new policy generation has been run.

## Code and verification evidence

- Resolver: `src/mechet/route_materials.py`; regression tests: `tests/test_route_materials.py`.
- Integration: `patches/route_material_boundary_20261116.patch`, against private worktree base `1b5ec65`; read-only final integration sources under `docs/audits/route_material_boundary_20261116/source/`.
- Deterministic replay entry point: `scripts/replay_route_material_boundary.py`; requires the private route runtime, saved TRAIN records and fixed stock, supplied through explicit arguments.
- Shared replay artifacts: `outputs/diagnostics/route_material_boundary_20261116_03/summary.json`. Reports containing original private actions stay on shared storage.
- Detailed test counts, command logs, review disposition and file bindings are recorded in the accompanying `verification.json` after final checks.

The public repository still does not contain the full private training stack. The integration patch and source copies are review artifacts, not a claim that the public checkout can independently run the complete private suite.
