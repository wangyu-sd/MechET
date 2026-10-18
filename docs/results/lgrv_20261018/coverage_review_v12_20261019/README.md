# Whole v12 coverage audit

| Stage | Unique complete reactions |
|---|---:|
| Frozen queries |18,210|
| Retained indexed candidates |165|
| Exact-candidate queries |49|
| Analogue-candidate queries |116|
| Supported among indexed queries |45|
| Indexed candidates still unresolved |120|
| No indexed candidate |18,045|
| Independently checked exact seeds among no-hit queries |2|
| Total supported |47(46exact+1analogue)|

45+120=165;45+2=47. Seeds are part of18,045 no-hit queries, not extra queries.
The remaining queue has5 exact and115 analogue queries. All18,210 audit rows
and every120 queued query remain in the distributed whole files, with original
source hashes in the [manifest](coverage_audit_manifest.json). The
[work-state inventory](../primary_review_states_v12_20261019/README.md)
distinguishes unread preparations from inspected-but-unresolved applicability.

The [full-index recall audit](../product_recall_v10_20261019/README.md) remains
unchanged:5,398 no-hit reactions have an identical product in the index,12,647
do not. Product identity alone is not reaction evidence; its absence does not
exclude a useful analogue or unindexed literature. Neither class is a count
of chemically incorrect predictions. Retrieval policy, all six conditions
and model predictions are unchanged.
