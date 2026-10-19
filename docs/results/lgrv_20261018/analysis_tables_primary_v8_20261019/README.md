# Complete v8 analysis: primary-source increment, not new predictions

All18,210 original unique generated reactions and all six frozen conditions
were reaggregated after nine additional directly inspected preparations.
The source corpus, query identities, context/hit row sequences, radius,
stereo policy and five-source cap are unchanged. No new model inference ran.

There are31 source-passage-checked exact experimental net reactions (TierA)
and18,179 unresolved reactions (TierC); no TierB or TierD was assigned.
The unresolved category is not a chemical-failure label. Nine is this review
increment, not the cumulative supported-reaction count.

| Frozen condition | Original targets | Observed reports | Source-supported complete reactions, all terminal contexts | Targets containing such a reaction |
|---|---:|---:|---:|---:|
| R1 MCTS |460|200|15|14|
| R1 Greedy I |460|460|4|4|
| R1 Greedy II |460|460|4|4|
| Single-reaction MCTS,10 total attempts |3,120|3,120|11|11|
| Single-reaction Greedy I |3,120|3,120|13|13|
| Single-reaction Greedy II |3,120|3,120|12|12|

Within R1's frozen stock-AND routes, MCTS retains199 nonzero stock closures,
9 targets with at least one exact-supported reaction and1 target with every
essential reaction supported. Each Greedy has zero targets with an exact
supported step on a stock-closed route; its supported reactions in other
terminal contexts are still counted above. Pending260 MCTS targets remain
pending, and the already-stock target is excluded from generated-route success.
The table's rows overlap chemically and must not be summed to obtain31.

Single-reaction outputs retain their original `rxn_prod_min` proxy DIAGNOSTIC
qualification. They are not corrected principal-product results or multistep
stock planning. Unequal archived work/completion permits recorded-run
description, not a matched-compute causal claim. In particular,11 versus13/12
is not evidence of MCTS superiority in that diagnostic.

`stored_artifacts_manifest.json` binds17 whole original analysis files; the
complete metrics JSON is losslessly gzipped, not sampled or summarized away.
`case_registry.csv` includes all31 supported reactions, source locators and
both separate method/target lists and exact paired `observed_memberships`.
`case_registry_manifest.json` binds its SHA to the actual v8 analysis manifest
`a1b11d5c6c44e009e79ac92c28003bb44bb711f835ffb99715dc09b5e515b497`.
Original large query/program inputs remain outside Git with frozen hashes.
Primary-source text is not redistributed wholesale; short cited witnesses and
local raw/text/normalization hashes retain the reproducibility chain.

Latest complete CPU software gate:1,253 passed/14 skipped, one existing PEFT
warning,170.06s. Software tests do not establish chemical accuracy.
