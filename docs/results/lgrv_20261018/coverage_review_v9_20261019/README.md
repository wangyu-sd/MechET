# Complete v9 retrieval/review missingness

All18,210 query-level audit rows and all131 unresolved indexed-candidate
review items are included. The source corpus and retrieval are unchanged.

| Coverage stage | Unique generated reactions |
|---|---:|
| Frozen queries |18,210|
| Queries with indexed structural candidates |165|
| Exact-candidate queries |49|
| Analogue-candidate queries |116|
| Source-supported among indexed queries |34|
| Indexed-candidate queries still unresolved |131|
| Queries without an indexed candidate |18,045|
| Independently checked seeds among no-hit queries |2|
| Total source-supported reactions |36|

The2 seeds are already part of18,045 no-hit queries, not extra queries.
Thus34+131=165 and34+2=36. The remaining queue contains15 exact-candidate
queries and116 analogue-candidate queries; three of the latter also have
separate source-inspected applicability notes, without a positive/negative
grade override. Source acquisition, original passage inspection and confirmed
substrate compatibility are distinct stages.

The previous full lookup-stage diagnosis remains applicable because query,
context and hit sequences are unchanged:17,553 no-hit queries have no current
exact/local-family lookup match,454 lack usable contexts,38 have raw local
candidates filtered by full context. This is not proof of absent literature.
TierC is unresolved, not invalid chemistry. No reference/NLL ranking or new
model inference was performed. Stored/original hashes are in the manifest;
the compressed JSONL files are whole outputs, not selected examples.
