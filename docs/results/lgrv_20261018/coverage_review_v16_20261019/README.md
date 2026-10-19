# Whole v16 coverage audit

| Stage | Unique complete reactions |
|---|---:|
| Frozen generated queries | 18,210 |
| Retained indexed-candidate queries | 165 |
| Exact-candidate queries | 49 |
| Analogue-candidate queries | 116 |
| Supported among indexed queries | 57 |
| Indexed-candidate queries unresolved | 108 |
| No indexed candidate | 18,045 |
| Independently checked seeds within the no-hit population | 2 |
| Total supported generated reactions | 59 (47 exact + 12 analogues) |

57 + 108 = 165; 57 + 2 = 59. The two seeds are part of the original 18,210
queries, not additional reactions. All queries and unsuccessful retrievals are
retained in the [manifest](coverage_audit_manifest.json). Two new analogue
supports follow individual original-document reviews, not a changed radius,
stereo policy, source cap, or similarity rule. All previous 57 support rows
remain unchanged.

The [work-state inventory](../primary_review_states_v16_20261019/README.md)
separates cached-but-unread preparations, checked applicability limits, and
source-side preparation/formulation limits. None is a chemical failure label.

The unchanged [whole product-index audit](../product_recall_v10_20261019/README.md)
finds 5,398 no-hit queries whose products occur in the source index, and 12,647
without an exact represented product. Product presence alone does not validate
precursor chemistry. A fixed-index no-hit is not an exhaustive literature
search or evidence that a generated reaction is impossible.

See the [complete v16 tables and all 59 cases](../analysis_tables_primary_v16_20261019/README.md).
No population chemical-accuracy, matched-compute, or microscopic-mechanism
validation is claimed. No new model/GPU/Taiji job was run.
