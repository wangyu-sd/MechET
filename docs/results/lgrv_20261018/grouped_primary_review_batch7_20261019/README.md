# Six primary preparations, twelve query-specific comparisons

Each of the six original capped candidates was reconciled against independently
transcribed named source structures. Complete referenced methods were read,
including second-level references. The tested compiler binds six preparations
and twelve source/query pairs: six compatible analogue judgements and six
unresolved applicability comparisons.

| Original source | Actual preparation record | Query-scoped outcome |
|---|---|---|
| US7465815B2, Example213 | BenzofuranylBr / dimethylaminopyridylboronic acid; 26%, mp/NMR; full107 method | Two compatible arylBr/boronic-acid analogues |
| US7632851B2, Example2 | Iodobenzothiadiazole / dimethylaminophenylboronic acid; orange powder, MS256; full1 method; no individual yield | Two compatible arylI/boronic-acid analogues |
| US8440652B2, 6.12 | Aminobromopyridine / carbamoylpyridylboronic acid; 35mg, 28%, NMR/MS410; full GeneralB | Two unresolved mesylate/polyamine/phenolic substrate settings |
| US8471005B2, 1043 | PyrrolotriazinylBpin / bromomethylimidazole; 16mg, 17%, NMR/MS389; full1042b | Two unresolved macrocyclic, opposite-handle settings |
| US8614226B2, 23 | Chloronaphthyridine / phenylpyridylboronic acid; 141mg, LCMS412; full20 workup | Two unresolved chloride activation/site-selection settings |
| US7951824B2, 139 | Iodopyridine / pyridylBpin; off-white solid, MS363.2/365.1; full131 step2 → 1 step6 | Two compatible arylI/Bpin analogues |

The positive judgements concern compatible net bond constructions, not measured
outcomes on the generated substrates. Every query has separate substrate,
functional-group, stereo, and condition-transfer limitations. No source yield
is assigned to a query and no MCTS-versus-Greedy advantage selects the cases.

Several source boundaries matter:

- The benzothiadiazole example has its own appearance/MS record; the generic
  preceding “can be prepared” introduction alone is not the evidence.
- GeneralB's scheme labels MeCN while its worked coupling uses nBuOH. Section
  6.12 does not restate its solvent; this ambiguity is retained.
- 1042a is Bpin precursor preparation, not 1043's coupling method. Likewise,
  131 step1's borylation is not139's coupling; the latter ultimately refers to
  Example1 step6, Pd(OAc)2/PPh3/carbonate, 90C/24h.
- Source22's conflicting product label and opposite handles are not substituted
  for the correctly matched23. Only20's workup, not its later deprotection and
  methylation, is referenced by23.

[Specification](../../../../configs/analysis/lgrv_grouped_primary_reviews_20261019_v7.json),
[bound audit](source_family_review.json), and
[six query-scoped positive records](../../../../configs/analysis/lgrv_primary_sources_20261019_v13.json)
are included. Reproduce with the unchanged
[grouped compiler](../grouped_primary_review_20261019/compile_reviews.py),
sealed v16 analysis, original traceback, and original primary cache.

These source-review results are **not yet integrated into a new pooled
analysis**. The complete published population remains v16 (59 supports).
No model/GPU task, new retrieval policy, radius/stereo/cap change, or independent
experimental calibration is claimed.
