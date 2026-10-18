# Five additional checked net transformations

These are actual frozen model-generated reactions with performed patent
preparations, not simulated predictions or invented literature. Independent
name-to-structure transcription was checked against the frozen exact reaction
key, without changing model products or precursor sets. The
[new primary registry](../../../configs/analysis/lgrv_primary_sources_20261018_v4.json)
contains original/cache URLs, exact raw/text hashes, short passage hashes,
source structures and preparation locators. The entire publication text stays
outside Git. Review was performed by the assistant, not a staffed human panel.

| Complete transformation | Performed primary preparation | Product status / source-specific limits |
|---|---|---|
| 4-Hydroxybenzoic acid + SOCl2 → 4-hydroxybenzoyl chloride | [US4645771](https://patents.google.com/patent/US4645771/en), Example25, starting-material preparation | Explicitly named crude residue carried into amidation; no isolated acid-chloride yield or characterization. |
| 3,5-Difluorobenzoic acid + SOCl2 → acid chloride | [US8629137B2](https://patents.google.com/patent/US8629137B2/en), Intermediate39, Step1 | Source reports1.45g/82% crude, not a purified isolated yield. Step2's93% is not transferred. Source acid mass/mmol are retained as printed. |
| o-Cresol + acetic anhydride → 2-methylphenyl acetate | [US5750050](https://patents.google.com/patent/US5750050/en), Step1.3 | Distilled product,83.3%; not the later Fries rearrangement product. |
| 5-Benzyloxy-2-pyridone + 2-bromothiazole carboxamide → N-arylated pyridone | [US8501746B2](https://patents.google.com/patent/US8501746B2/en), Example3.2, following Example3 |32% isolated, melting point/NMR/MS; CuI/8-hydroxyquinoline/K2CO3/DMSO conditions. It is C–N coupling, not O-arylation or a demonstrated catalyst-free reaction. |
| 3-(1-Cyano-1-methylethyl)benzoic acid + SOCl2 → acid chloride | [US8344135B2](https://patents.google.com/patent/US8344135B2/en), ExampleD25(i), first paragraph | Explicitly named colorless oil, subsequently used; no independent yield or purity. Do not add the ring chlorine found in other examples. |

All five exact keys, cache receipts, HTML-to-text normalization and declared
passage offsets passed actual checks. They add **five TierA net-reaction
precedents**, with no blanket TierB approval. All six frozen conditions were
reaggregated in the separate completed `source_checked_20261018_v5` output.

Across all18,210 reactions, checked exact precedents increased4→9. R1 MCTS
targets with any exact step on a stock-closed route increased2→5; the
every-essential-step-supported route-target count remains1. Its stock closure
count remains199/460 with260 reports pending. Greedy route counts and all
original denominators are unchanged. These are **evidence-discovery counts**,
not a population chemical-accuracy estimate or an equal-compute advantage.
The complete [v5 tables](analysis_tables_primary_v5_20261018/README.md) include
all successes and unresolved cases, all target pairings and all baseline runs.

The source support concerns each full net transformation. It does not
experimentally validate each symbolic electron arrow, the actor's auxiliary
mixture or an entire generated route. Source/training overlap remains unknown.
