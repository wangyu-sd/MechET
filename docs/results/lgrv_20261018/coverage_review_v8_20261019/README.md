# Full v8 retrieval and source-review coverage

This directory preserves the whole18,210-query audit and all136 remaining
indexed-candidate review items, not a selected positive subset.

| Stage | Unique generated reactions |
|---|---:|
| Frozen queries |18,210|
| At least one indexed structural candidate |165|
| Exact-candidate queries |49|
| Analogue-candidate queries |116|
| Checked support among indexed queries |29|
| Indexed candidates still unresolved |136|
| No indexed candidate |18,045|
| Independently source-checked seeds among no-hit queries |2|
| Total source-checked supported reactions |31|

The two seeds are included in18,045 no-hit queries, not additional query rows.
Thus165+18,045=18,210;29+136=165;29+2=31. Source-supported31 includes the latest
nine additional preparations, not just nine cases overall. The corpus is the
same frozen93-file collection; no claim of exhaustive reaction-literature
coverage is warranted. Lack of an index hit is not proof of absent precedent.

Native query context remains available for17,755 reactions;354 have no
reaction center and101 conflicting native families. Of no-hit queries,17,591
still have an available context. Therefore missing native context alone does
not explain the broad retrieval gap. All unknown states stay in the audit.

The compressed JSONL files contain every query-level coverage row and every
remaining candidate-review item. Their original paths/hashes are in
`stored_artifacts_manifest.json`; `coverage_audit_manifest.json` binds the
actual v8 analysis and all four input tables. A TierC entry means unresolved,
not chemically wrong. These counts measure retrieval/review coverage, not
endpoint accuracy, universal mechanism validity or a source-supported yield.
