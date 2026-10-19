# Whole v18 coverage audit

| Stage | Unique complete reactions |
|---|---:|
| Frozen generated queries |18,210|
| Retained indexed-candidate queries |165|
| Exact-candidate queries |49|
| Analogue-candidate queries |116|
| Supported among indexed queries |71|
| Indexed-candidate queries unresolved |94|
| No indexed candidate |18,045|
| Independently checked seeds within the no-hit population |2|
| Total supported generated reactions |73 (47 exact +26 analogues)|

71+94=165;71+2=73. Both seeds are part of the original18,210 queries,not
additional reactions. The [manifest](coverage_audit_manifest.json) binds all
queries and failed retrievals. All73 grades/selected sources and complete
query/context records are unchanged versus v17; nine candidate lists are
family-deduplicated. The earlier v16-to-v17 source review added14 analogue
queries,not this bibliographic correction.

The [review-state inventory](../primary_review_states_v18_20261019/README.md)
separates33 unresolved original-preparation pointers,50 checked applicability
limits,9 preparation/formulation limits and2 stereo limits. No original
cached query remains unread. These are evidence states,not chemical failures.

The unchanged [whole product-index audit](../product_recall_v10_20261019/README.md)
finds5,398 no-hit queries whose products occur in the source index,and12,647
without an exact represented product. Product presence alone does not validate
the precursor transformation. A no-hit in this frozen index is neither an
exhaustive literature search nor evidence that the generated reaction cannot
occur. Retrieval-policy/source limitations remain explicit.

See [all73 cases and whole tables](../analysis_tables_primary_v18_20261019/README.md).
No population chemical accuracy,matched-compute advantage or microscopic
mechanism validation is inferred. No model/GPU/Taiji job ran.

Bibliographic correction: the full v18 rerun uses original-page patent-family
identities before the source cap. Candidate hits fall from210 to198 across
nine query lists; all165 indexed-query IDs, all73 grades/selected sources and
all six route/composition/paired results are unchanged. This is not a new
model experiment or a chemical-success estimate. See the
[whole-population correction audit](../family_dedup_review_v18_20261019/whole_population_audit.json).
