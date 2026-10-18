# Whole v15 coverage audit

| Stage | Unique complete reactions |
|---|---:|
| Frozen generated queries |18,210|
| Retained indexed-candidate queries |165|
| Exact-candidate queries |49|
| Analogue-candidate queries |116|
| Supported among indexed queries |55|
| Indexed-candidate queries unresolved |110|
| No indexed candidate |18,045|
| Independent checked seeds among those no-hit queries |2|
| Total supported generated reactions |57(47exact+10analogues)|

55+110=165;55+2=57. The seeds belong to the18,045 no-hit population,not
additional queries. All18,210 audit rows andall110 pending source-evidence
rows are preserved in the [manifest](coverage_audit_manifest.json).

The three new analogue supports follow individually authored reviews of
10 source/query pairs,not a new retrieval or similarity rule. Query structures,
radius,stereo andcap5 remain unchanged. The [work-state inventory](../primary_review_states_v15_20261019/README.md)
distinguishes unread preparations from checked but unresolved applicability.
The unchanged [full product-index audit](../product_recall_v10_20261019/README.md)
shows5,398 no-hit reactions with represented products and12,647 without an
exact represented product. Neither group is a chemical error count.
