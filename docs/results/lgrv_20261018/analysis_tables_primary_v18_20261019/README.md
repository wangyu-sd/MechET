# Complete v18 literature-evidence tables

All18,210 frozen complete-reaction queries and six recorded conditions are
retained: **47 exact precedents,26 query-scoped analogues,18,137 unresolved**.
The14 supports added in the historical v16-to-v17 update are precisely the source-checked queries in the
[completed cached-review queue](../SOURCE_REVIEW_COMPLETION_20261019.json).
Compared with v17,all73 grades and selected sources,all query/context records
and all aggregate scientific results are unchanged; nine candidate lists are
family-deduplicated. This is a bibliographic correction,not retraining or a
model improvement.

The73 supported generated reactions are not73 independent experiments or a
population chemical-accuracy estimate. A/B support refers to the complete net
transformation; ordinary patent preparations do not establish each internal
FLOW arrow. The actual recorded query substrates remain unmeasured for TierB.

| R1 recorded condition | Saved reports / fixed targets | Distinct supported reactions | Targets with support | Nonzero stock-AND closures |
|---|---:|---:|---:|---:|
| MCTS |200/460|35|28|199|
| Greedy I |460/460|7|6|38|
| Greedy II |460/460|7|6|36|

Reaction/target support here includes all terminal contexts, not only edges
on stock-closed routes. On such routes,MCTS retains11 targets with an exact
supported step and one with every essential step supported. Neither outcome
changes from v17. The260 unsaved MCTS reports remain pending,not failures;
the separate already-stock target is not a newly generated synthesis route.

| 31k frozen proxy condition | Fixed targets | Recorded budget | Distinct supported reactions | Targets with support | Exact-supported targets |
|---|---:|---|---:|---:|---:|
| MCTS |3,120|10 total trajectory attempts|32|30|22|
| Greedy I |3,120|One greedy episode|18|18|18|
| Greedy II |3,120|One greedy episode|17|17|17|

This remains the frozen `rxn_prod_min` proxy diagnostic,not corrected
principal-product prediction,stock-route planning,NLL-ranked Top-K or
reference-endpoint accuracy. Different recorded work/completion budgets make
method comparisons descriptive,not matched-compute causal gains. The73 pooled
reactions are chemically deduplicated across methods; do not sum table rows
to reconstruct that numerator.

R2's source-supported seen/unseen program reaction sets are28/11 and overlap.
Eleven targets have an unseen whole program leading to a supported reaction;
one has a precedent-covered route with the recorded unseen-program witness.
This is not a new-chemistry claim,reference C1/C2 label or strict-H2 result.

All22 sealed analysis outputs,current producer hashes and22 positive-source
registries are authenticated. All22 sealed outputs plus the analysis manifest are byte-identical
or losslessly gzipped copies bound by the
[distribution manifest](stored_artifacts_manifest.json). The
[73-case registry](case_registry.csv) contains every supported reaction,
structures,source locator/document hashes and exact run/target memberships;
no method-favouring case selection is applied.

The [whole coverage audit](../coverage_review_v18_20261019/README.md),
[94-query review-state inventory](../primary_review_states_v18_20261019/README.md),
and [six-condition summary](../POPULATION_RESULTS_PRIMARY_V18_20261019.json)
retain the same populations. Source reviews:

- [Six coupling preparations](../grouped_primary_review_batch7_20261019/README.md).
- [Four substitution preparations](../grouped_primary_review_batch8_20261019/README.md).
- [Seven further coupling preparations](../grouped_primary_review_batch9_20261019/README.md).
- [Thirteen additional preparations](../grouped_primary_review_batch10_20261019/README.md).
- [Preparation/stereolabel limits](../source_preparation_limits_batch2_20261019/README.md).
- [O/C-ethylation and source-formulation limits](../source_preparation_limits_batch3_20261019/README.md).

Reproduction uses the unchanged public
[`verify_nmi_literature_reactions.py`](../../../../scripts/verify_nmi_literature_reactions.py),
the [frozen configuration](../../../../configs/analysis/nmi_r1_r2_matched_baselines_20261017.yaml),
the22-output analysis manifest's input hashes and all22
`primary_registry_manifests` paths. Supply the original prediction freeze,
program join,native mapping and independent-policy-control roots listed in
the parent results documentation,then choose a **new** output directory.
The persistent source index is
`data/derived/literature/ord_lgrv_full_20261018_v1/reactions.sqlite` in the
main checkout,SHA256
`ee997c8732df797e27a7d5c2bd58cdd18a4472566818f9d8870092ba4f0de16b`.
This run used its physically rehashed byte-identical local copy for transport
only; full query/context records and every pre/post candidate list were
independently compared with v17,including the nine changed lists.
The temporary copy is not required for reproduction.

No new model/GPU/Taiji job,reaction-identity/radius change or main/paper merge ran.
Population figures and final whole-goal review are separate delivery gates;
the whole PR89 scope is not declared complete by these tables alone.

Bibliographic correction: the full v18 rerun uses original-page patent-family
identities before the source cap. Candidate hits fall from210 to198 across
nine query lists; all165 indexed-query IDs, all73 grades/selected sources and
all six route/composition/paired results are unchanged. This is not a new
model experiment or a chemical-success estimate. See the
[whole-population correction audit](../family_dedup_review_v18_20261019/whole_population_audit.json).
