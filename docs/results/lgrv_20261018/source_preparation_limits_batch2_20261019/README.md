# A stereolabel or named product is not a performed transformation

Three original source/candidate pairs for two generated queries retain Tier C:

| Primary source | What is actually reported | What is not established |
|---|---|---|
| US7745646B2, 61 / 52 | Acetate product NMR/MS; conditional KOAc preparation;52's actual methylpiperazine substitution | Explicit execution of the indexed KOAc recipe; transfer to a different intramolecular cyclic-acetal closure |
| US6858642B1, 5/6/Table1 | Caffeic-acid amide/ester preparations and an antioxidant-control comparison | A one-component unspecified-alkene → E-caffeic-acid reaction |
| US7094801B2, 81/88 | Actual HCl salt formation, 54mg/78%; the parent81 is already named3E | Newly generated E geometry in88, or precedent for the query's charged sulfur/nitrile substrate |

Product existence and characterization are not denied. In particular,88 is an
actual preparation, but its indexed input lacks the stereo label already in
81's title. That absence is not proof of a Z starting material or an observed
isomerization. Its neutral-parent-plus-Cl encoding is retained, not silently
converted to an ionic salt. Example81's repeated precursor-name issue is also
not reconstructed. In the coumarin patent, the “isolated during purification
of58” sentence belongs to59, not the acetate61.

[Specification](../../../../configs/analysis/lgrv_source_preparation_limits_20261019_v3.json)
and [bound audit](source_limits_review.json) include original structures,
unique short witnesses, source limits, and query-specific interpretations.
No positive registry or chemical-failure label is emitted.

## Versioned lookup-address repair

The original index retains zero-padded patent IDs such as `US06858642B1`.
The existing acquisition module derives the public lookup `US6858642B1`
without inventing a publication kind or family. The old limit binder compared
the URLs literally and rejected that same-source pairing. The new versioned
[binder](bind_limits.py) uses that existing validated lookup mapping and saves
both original and lookup IDs/URLs in each direct-source binding. Original
candidate IDs, chemistry, source cap, raw source files, and old sealed recipes
are unchanged. It cannot turn a source review into positive evidence.

The actual old-reader failure was reproduced by a real-file regression before
the fix. Eight new software tests cover the valid padded alias plus rejection
of wrong publication kind/number, mismatched ID/URL, untrusted host, wrong
candidate structure, altered primary bytes, and prior-support downgrade.
The old and new limit-binder tests pass together (18 tests); these are software
checks, not model experiments.

```bash
PYTHONPATH=src python docs/results/lgrv_20261018/source_preparation_limits_batch2_20261019/bind_limits.py \
  --review-spec configs/analysis/lgrv_source_preparation_limits_20261019_v3.json \
  --analysis-root /path/to/source_checked_20261019_v16 \
  --traceback docs/results/lgrv_20261018/coverage_traceback_20261018/candidate_patent_traceback.jsonl.gz \
  --cache-root /path/to/primary_documents_bulk_20261018_v2 \
  --output /new/path/source_limits_review.json
```

These new source findings await pooled accounting. Published v16 tables and
figures retain their original59-support counts. No model/GPU task, altered
prediction, new source candidate, or full-patent redistribution occurred.
