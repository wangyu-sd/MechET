# Complete v10 retrieval and review missingness

All18,210 query-level audit rows and all129 unresolved indexed-candidate
review items are included. Corpus, queries, matching rules and cap5 are unchanged.

| Coverage stage | Unique generated reactions |
|---|---:|
| Frozen queries |18,210|
| Queries with indexed structural candidates |165|
| Exact-candidate queries |49|
| Analogue-candidate queries |116|
| Source-supported among indexed queries |36|
| Indexed-candidate queries still unresolved |129|
| Queries without an indexed candidate |18,045|
| Independently checked seeds among no-hit queries |2|
| Total source-supported reactions |38|

The2 seeds are included in18,045 no-hit queries, not extra queries.
Thus36+129=165 and36+2=38. The queue has13 exact-candidate and116
analogue-candidate queries. Known source-inspected unresolved cases remain in
that queue. They are not automatically positive, negative or unreviewed.

The [whole review-state inventory](../primary_review_states_v10_20261019/README.md)
identifies92 queries with cached originals awaiting preparation review,
33 without a recovered original-preparation pointer,3 inspected but
applicability-unresolved analogue cases, and1 partially inspected positional-
isomer mismatch with other candidates still pending. All90 referenced primary
documents are locally available; availability does not imply chemical support.

The unchanged full lookup audit gives17,553 no-hit queries without a current
exact/local-family key candidate,454 without usable contexts and38 with raw
local candidates filtered by context. This is not proof of absent literature.
TierC means unresolved, not invalid chemistry. Stored/original hashes are in
the manifest; these are whole outputs, not selected examples. No new model
inference or reference/NLL ranking was performed.
