# Whole-index product-identity recall diagnosis

This read-only audit canonicalizes the product of **every4,338,961 source
representation** in the frozen index and compares it to all18,210 generated
reaction queries (6,675 distinct products). It uses the same map-free,
isotope/charge/stereo-preserving component multiset, without desalting,
neutralization, tautomer changes or ignoring component multiplicity.

| Retained reaction candidate | Same complete product identity in index | Generated reaction queries |
|---|---|---:|
| No |No|12,647|
| No |Yes|5,398|
| Yes |Yes|99|
| Yes |No|66|
| Total | |18,210|

The first two rows sum to18,045 original no-hit queries. The last two sum to
165 original retained-candidate queries. A retained analogue may have a
different product, explaining the66 in the last row. The index contains8,371
representation rows with the same product identity as at least one query;
representation rows are not unique experiments or independent sources.

This narrows the earlier lookup audit:5,398 no-hit reactions have a product
represented in the source collection, but no candidate under the frozen
complete-reaction/local-context retrieval. That can reflect genuinely different
precursors or incompatible transformations, not necessarily a lookup bug.
For12,647 no-hit reactions, even the conservative exact product is absent.
This can reflect source coverage or representation conventions; it does not
exclude valid analogues or unindexed experimental literature.

**Same product is not reaction evidence.** These rows do not validate generated
reactants, conditions or electron moves, add primary candidates to the formal
cap5 queue, change any grade, or establish chemical accuracy. All current38
source-supported reactions and six-condition results remain unchanged. This
diagnostic is not a retuned test-set retrieval policy.

`product_recall_queries.jsonl.gz` retains all18,210 query identities, the
original retained-hit flag, product-identity counts and distinct source-reaction
key counts. `product_matching_source_rows.jsonl.gz` retains **every8,371**
matching representation identity and source reaction key, not selected
examples. The manifest binds all output bytes, original analysis inputs,
the full index SHA and the byte-identical recipe. Both full-index checksum
passes, source product parsing and all joins completed; no source product was
silently omitted as invalid.

Reproduce with the original frozen index and repository RDKit environment:

```bash
PYTHONPATH=.:src python docs/results/lgrv_20261018/product_recall_v10_20261019/audit_recipe.py \
  --analysis-root /path/to/source_checked_20261019_v10 \
  --output-root /new/path/to/product_recall \
  --workers 8
```

Only output bytes are written; the source database is opened immutable/read-only.
No new model/GPU job or reaction sampling was performed.
