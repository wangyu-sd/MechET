# Three performed preparations, eighteen explicit source/query reviews

One source preparation can be relevant to several generated reactions,but its
compatibility is **not inherited across a local reaction family**. This batch
directly inspects the three original preparations below and records every
one of their18 pending source/query pairs:6 compatible analogues,12 unresolved.
These are assistant original-source judgements,not new experiments or an
independent policy-calibration set. No similarity threshold,radius,candidate
cap,query set or model prediction changed.

| Performed preparation | Pairs reviewed | New TierB | Retained TierC |
|---|---:|---:|---:|
| US7855213B2,Example85;Example78 method |7|2|5|
| US8247576B2,Example55;Example1 method |7|4|3|
| US4471125,Example1 |4|0|4|
| Total |18|6|12|

The first source reports16%yield,not Example78's13.4%. The second reports
NMR/LCMS but no individual yield;Example1's9% is not transferred. The third
reports45.6% **analytical** yield under250 C solvent-free conditions,not a
specified isolated mass. Its printed1.50g/76.7mmol is flagged as internally
inconsistent rather than silently repaired. PubChemCID246078 is used only to
resolve conventional phthalimide nomenclature,not as reaction evidence.

The6 TierB judgements preserve the reaction-center/local-stereo guards and
explicitly describe changes in coupling partner,remote heterocycles,alcohol,
carbamate,carbonyl or configured amino side chain. The source is a performed
related coupling;query conversion,selectivity andyield remain unmeasured.
They do not establish the model's exact auxiliary mixture or electron arrows.

The12 unresolved pairs have concrete limits:acid-chloride survival in aqueous/
alcoholic medium;remote alkene/halide selectivity;charged sulfonamide formulation;
competing aminoalkyl-bromide chemistry;activated heteroaryl chloride/boronate
speciation;multiple competing donors;or transfer from activated phthalimide
substitution to a differently activated chloroarene. These are reasons this
particular source does not establish compatibility,**not observed failed
reactions or TierD contradictions**. Alternative conditions may work.

## Files and reproducibility

- [Explicit common preparations and all18 individual reviews](../../../../configs/analysis/lgrv_grouped_primary_reviews_20261019_v1.json).
- [Bound source/pair audit](source_family_review.json):quotes appear once per
  common preparation;raw/text/normalization/locators,independent named source
  graphs and every B/C decision remain.
- [Positive query-scoped registry](../../../../configs/analysis/lgrv_primary_sources_20261019_v8.json):only6 individually accepted pairs. No blanket family rule.
- [Compiler](compile_reviews.py):rehashes every sealed frozen output and
  original pointer traceback,rebinds source preparations,checks exact source
  structures against an original capped candidate,and invokes the existing
  evidence checker on each query's full-stereo context. It cannot invent the
  compatibility judgement;the reviewed specification supplies it.
- [All6 source/query comparisons](../analogue_source_query_gallery_20261019_v14/README.md)
  distinguish original experiments from generated analogue hypotheses.

Eight synthetic CPU software tests cover per-query support,unresolved peers,
corrupt frozen/source bytes,wrong candidate identity,unrelated primary pointer,
duplicate review pairs,unsupportedB flags andpopulation mismatch. They are not
experimental chemistry results. Full CPU suite:1,269passed/14skipped,one
existing PEFT warning,258.93s.

```bash
PYTHONPATH=src python docs/results/lgrv_20261018/grouped_primary_review_20261019/compile_reviews.py \
  --review-spec configs/analysis/lgrv_grouped_primary_reviews_20261019_v1.json \
  --analysis-root /path/to/source_checked_20261019_v13 \
  --traceback docs/results/lgrv_20261018/coverage_traceback_20261018/candidate_patent_traceback.jsonl.gz \
  --cache-root /path/to/primary_documents_bulk_20261018_v2 \
  --audit-output /new/path/source_family_review.json \
  --registry-output /new/path/primary_registry.json
```

No model/GPU task ran andno raw patent full text is distributed here.
