# Complete v13 retrieval and source-review coverage

| Stage | Unique complete reactions |
|---|---:|
| Frozen generated queries |18,210|
| Retained indexed-candidate queries |165|
| Exact-candidate queries |49|
| Analogue-candidate queries |116|
| Source-supported among indexed queries |46|
| Indexed-candidate queries unresolved |119|
| No indexed candidate |18,045|
| Independently checked seeds among those no-hit queries |2|
| Total source-supported |48(47exact+1analogue)|

46+119=165;46+2=48. Seeds remain part of the no-hit population,not additional
queries. The whole remaining queue contains4 exact and115 analogue queries.
All18,210 audit rows and all119 queued reactions are retained with original
input/output hashes in the [manifest](coverage_audit_manifest.json).

[Work-state inventory](../primary_review_states_v13_20261019/README.md)
separates unread preparations,unresolved original pointers and inspected
applicability/stereochemistry limits. Frozen retrieval remains unchanged;
the existing [whole-index product audit](../product_recall_v10_20261019/README.md)
still distinguishes5,398 no-hit reactions with a product represented in the
index from12,647 without. Neither category is a chemical error count.
