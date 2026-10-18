# Why most frozen reactions have no indexed candidate

This is a full18,210-query, read-only lookup-stage audit of the actual frozen
18GB source index and v8 queries/contexts/hits. It changes no retrieval rule,
reaction identity, source cap, model prediction or evidence grade.

| Mutually exclusive lookup stage | Unique reaction queries |
|---|---:|
| Retained indexed candidate |165|
| No exact-key or current legacy-family-key candidate, with an available context |17,553|
| No hit and unavailable reaction-center context |454|
| Raw legacy-family candidates exist, but none survive the current full-context check |38|
| Total |18,210|

The last three rows total18,045 no-hit queries. The454 unavailable-context
queries are354 without a center and100 with conflicting native families;
one additional conflicting-family query still has an exact hit and belongs
to the165 retained queries. Exact lookup remains available without a usable
analogue context. No query with actual exact index rows lost all its hits.

Thus the dominant observed bottleneck is **no candidate under the existing
lookup keys**, not primary-passage grading or the later compatibility filter.
Removing the latter could affect these38 queries, not explain the17,553.
It would not automatically make the rejected candidates chemically compatible.
All38 filtered queries were additionally replayed through the unchanged
production `query_sources(..., cap=5)` function: all173 raw representation
records were rejected by its full-context check, and none returned a hit.
Absence under these exact/local keys does not mean absent literature: it can
reflect the indexed source domain, structure/role conventions or a too-specific
reaction-center description. This audit does not distinguish those causes and
does not establish universal analogue recall or chemical validity.

`query_lookup_bottlenecks.jsonl.gz` retains every query, its exact/raw-family
representation-row counts, context status and original retained-hit count.
Raw SQL row counts include representation aliases and are not distinct
experiments. The manifest binds the original index SHA, all four input tables,
the v8 analysis SHA, complete output hash and byte-identical `audit_recipe.py`.
The recipe authenticates the index before immutable read and checks original
input hashes again before sealing the output. It never writes the source DB.

Reproduction, with the repository Python environment and RDKit available:

```bash
PYTHONPATH=.:src python docs/results/lgrv_20261018/lookup_bottlenecks_v8_20261019/audit_recipe.py \
  --analysis-root /path/to/source_checked_20261019_v8 \
  --output-root /new/path/to/lookup_bottlenecks
```

The original analysis's source-index path must be accessible; the recipe is
pinned to the actual v8 manifest, not an arbitrary replacement result set.
Source-supported31 and all existing source/route metrics remain unchanged.
