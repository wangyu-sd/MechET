# Complete v11 retrieval and review coverage

All18,210 query audit rows and every123 unresolved indexed-candidate query
are retained. Retrieval policy and original model-generated predictions are
unchanged. Acquiring a source is not the same as verifying its preparation.

| Coverage stage | Unique generated reactions |
|---|---:|
| Frozen queries |18,210|
| Queries with indexed structural candidates |165|
| Exact-candidate queries |49|
| Analogue-candidate queries |116|
| Source-supported among indexed queries |42|
| Indexed-candidate queries still unresolved |123|
| Queries without an indexed candidate |18,045|
| Independently checked seeds among no-hit queries |2|
| Total source-supported reactions |44|

The2 seeds are included in18,045 no-hit queries, not extra queries.
Thus42+123=165 and42+2=44. The remaining queue contains7 exact and116
analogue queries. The [whole work-state inventory](../primary_review_states_v11_20261019/README.md)
separates85 cached/preparation-pending,33 original-pointer-unresolved,
3 checked/applicability-unresolved,1 checked/stereo-identity-unresolved and
1 partially checked isomer discrepancy. All82 distinct original documents
referenced by the remaining queue are available; that does not confer support.

The unchanged full lookup diagnosis and [whole-index product audit](../product_recall_v10_20261019/README.md)
still apply:of18,045 no-hit queries,5,398 have an exact product represented in
the index and12,647 do not. A product match does not verify reactants or
conditions; product absence does not exclude analogues/unindexed literature.
TierC remains unresolved, never automatically invalid. Whole output hashes
and original input paths are in the manifest; no examples are substituted for
the full population and no new reference/NLL ranking was performed.
