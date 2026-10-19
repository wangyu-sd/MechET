# Four substitution precedents and their substrate limits

All four original candidate preparations were directly read and independently
transcribed. The unchanged grouped-review compiler authenticates four pairs:
two compatible benzylic substitutions and two unresolved applicability cases.

| Primary preparation | Source observation | Generated-query judgement |
|---|---|---|
| US6579884B1, 1.56 | HeterobenzylicCl + sodium methanethiolate; RT/MeCN/4h; 50mg, 24%, NMR | Compatible analogue for3-bromobenzylCl + methanethiolate |
| US4401598, 8 | BenzylicCl + NaN3; defined quaternary-ammonium catalyst; product bp/IR/elemental analysis; 89 **weight** percent | Compatible analogue for chloro/cyanobenzylCl + azide |
| US4363805, 41 / 38(a) | Sodium phenolate + 1,8-dibromooctane; isolated bromooctyl ether, 75% | Unresolved transfer to an NH-lactam phenoxide / shorter dibromopropane |
| US7601736B2, 16 | NaOMe + chlorodimethylnicotinonitrile; 2.41g, 99%, NMR | Unresolved retained protonated-nitro form and chloride-site choice |

The source sodium counterions remain part of the source structures. Missing
query counterions are not inserted. The azide patent's printed weight percent
is not recast as a molar isolated yield, and other examples' solvents are not
silently borrowed. In16, 0C is the addition temperature; the subsequent hold
temperature and NaOMe purity are not reconstructed from another example.

The unresolved cases are not documented chemical failures: the source does not
settle the added NH/short-tether selectivity or the supplied charged nitro-like
state. No functional group, proton, charge, or leaving-group site is changed
to make a query fit.

[Specification](../../../../configs/analysis/lgrv_grouped_primary_reviews_20261019_v8.json),
[bound audit](source_family_review.json), and
[two query-scoped positive records](../../../../configs/analysis/lgrv_primary_sources_20261019_v14.json)
are included. Reproduce with the unchanged
[grouped compiler](../grouped_primary_review_20261019/compile_reviews.py),
sealed v16 analysis, original traceback, and primary cache. These judgements
await the next pooled refresh; published full-population v16 counts remain
unchanged. No model/GPU inference or chemistry experiment ran.
