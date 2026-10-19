# Why nine checked reactions did not describe population chemistry

This audit covers **all 18,210 frozen unique generated reactions**. It diagnoses
retrieval and source-review missingness; it does not regrade chemistry, rerun
the actor, add chemical candidates, or change the frozen radius/source cap.

| Archived v5 stage | Distinct reactions |
|---|---:|
| All queries |18,210|
| At least one indexed candidate |165|
| Exact candidate |49|
| Analogue candidate |116|
| No indexed candidate |18,045|
| Source-supported, with an indexed candidate |7|
| Source-supported, independently checked seed outside the hits |2|
| Candidate present, primary support unresolved |158|

The exact/analogue counts happen not to overlap in this snapshot. Source-record
counts are different units:92 exact and118 analogue records,210 in total.
The nine checked transformations were therefore **not** nine chemically valid
reactions out of18,210. They were an incomplete source-review lower bound.

Of18,045 no-hit reactions,17,591 have an available stereo-aware reaction centre,
354 have no bond/charge centre and100 have conflicting native families. Missing
mapping alone cannot explain the low hit rate. The audit does not establish
whether an individual no-hit is absent from the corpus, mismatched in structural
roles/charge, or chemically implausible. These possibilities need separate tests;
unknown reactions remain unknown.

## Bibliographic provenance recovery

All139 non-patent records point to **one** DOI,10.1039/C8SC04228D. This is
[Coley et al.'s reaction-prediction paper](https://pubs.rsc.org/en/content/articlehtml/2019/sc/c8sc04228d),
listed by the [official ORD catalogue](https://github.com/open-reaction-database/ord-data/blob/main/README.md)
as a source of training/test data. It is not139 independent experimental papers.
Its presence in a mined record does not identify a performed preparation.

For each such candidate, the new audit looks up the **candidate's own exact
complete reaction key**, not the generated query's key. This matters for
analogues: another substrate's patent must never become an exact precedent for
the model substrate. Original source IDs, URLs, query/candidate keys and match
types are retained. No publication kind or patent family is inferred.

This recovered alternate patent pointers for106/139 candidate records,176
pointer occurrences and104 distinct normalized public patent URLs. All104
documents were cached and their raw/text checksums verified. Compared with the
previous66-document collection,71 are newly available,33 overlap and the union
contains137 distinct patent pages. **These are document-acquisition counts,
not additional supported reactions.** The remaining33 data-set records are
still unresolved at this traceback step.

The chemical candidate pool remains the original165 queries/210 records. The
additional pointers are bibliographic representations of those same candidates,
not extra independent candidates or a changed source-selection policy. Any new
positive grade still needs inspection of the particular performed experiment,
structural reconciliation and the frozen per-query verification policy.

## Stored evidence

- `query_coverage_audit.jsonl.gz`: all18,210 queries, including every no-hit.
- `primary_review_queue.jsonl.gz`: all158 unresolved v5 candidate queries,
  with exact-example versus analogue-review actions distinguished.
- `candidate_patent_traceback.jsonl.gz`: all139 non-patent candidates, including
  unsuccessful tracebacks, with the recovered patent pointers.
- `patent_traceback_acquisition_references.jsonl.gz`: all176 alternate
  bibliographic references, not new generated chemical predictions.
- `traceback_acquisition_manifest.json.gz`: lossless complete104-document
  acquisition receipt, not a selected summary. Full patent text stays outside Git.
- `stored_artifacts_manifest.json`: original paths and exact byte hashes.

The separate [v6 full reaggregation](../analysis_tables_primary_v6_20261018/README.md)
adds six checked preparations and reports15 TierA reactions. The v5 audit and
its158-entry queue are preserved rather than silently relabelled as v6.

Reproduce the audit from the authenticated full analysis:

```bash
PYTHONPATH=src python -m mechet.analysis.literature_coverage_audit \
  --analysis-root <full_source_checked_v5_root> \
  --output-root <new_output_root> --traceback-patents
```

Software verification: seven regression tests cover complete populations,
seed-without-hit support, citation deduplication, exact-candidate traceback,
analogue-status retention and corrupted-input rejection. These are software
tests, not model-performance experiments.
