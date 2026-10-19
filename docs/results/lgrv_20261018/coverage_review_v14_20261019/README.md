# Whole v14 coverage audit

| Stage | Unique complete reactions |
|---|---:|
| Frozen generated queries |18,210|
| Retained indexed-candidate queries |165|
| Exact-candidate queries |49|
| Analogue-candidate queries |116|
| Supported among indexed queries |52|
| Indexed-candidate queries unresolved |113|
| No indexed candidate |18,045|
| Independent checked seeds among those no-hit queries |2|
| Total supported generated reactions |54(47exact+7analogue)|

52+113=165;52+2=54. Seeds are part of the18,045 no-hit queries,not additional
queries. All4 remaining exact-candidate and109 analogue-candidate queries
are kept in the review queue. All18,210 audit rows,all113 queued rows and
source hashes are in the [manifest](coverage_audit_manifest.json).

The [work-state inventory](../primary_review_states_v14_20261019/README.md)
distinguishes unread preparations from checked applicability/stereo limits.
The [full product-index audit](../product_recall_v10_20261019/README.md)
remains unchanged:5,398 no-hit reactions have a product represented in the
index,12,647 do not. Neither class is a chemical error count. Candidate
radius,stereo rules,cap5 andall predictions are unchanged;source review
does not retroactively become an independent policy calibration.
