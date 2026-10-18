# Complete v11 source evidence, unchanged frozen predictions

Six further original performed preparations were read and independently
structurally transcribed. Each referenced general method was also checked;
individual yields, salt assignments and conditions were not borrowed from
different examples. All18,210 unique reactions, six conditions, corpus,
stereo/local rules and cap5 remain unchanged. All21 sealed output hashes and
complete original query/context/hit sequences were verified against v10.

There are44 source-passage-checked exact net reactions (TierA),18,166
unresolved (TierC), and no TierB/D. This is source-review coverage, not
population chemical accuracy or experimental verification of symbolic arrows.

| Frozen condition | Original targets | Observed reports | Supported reactions across all terminal contexts | Targets with such a reaction |
|---|---:|---:|---:|---:|
| R1 MCTS |460|200|17|16|
| R1 Greedy I |460|460|6|6|
| R1 Greedy II |460|460|6|6|
| Single-reaction MCTS,10 total attempts |3,120|3,120|21|21|
| Single-reaction Greedy I |3,120|3,120|18|18|
| Single-reaction Greedy II |3,120|3,120|17|17|

Reaction identities overlap across methods, so rows cannot be summed to44.
R1 MCTS has199 nonzero frozen-stock AND closures,11 targets with an exact-
supported reaction on a stock-closed route, and1 with every essential step
supported. Each Greedy has zero supported-step stock-closed targets; its six
supported reactions in other contexts remain included. The260 pending MCTS
targets stay pending; the already-stock target is not newly solved.

Single-reaction results remain the frozen `rxn_prod_min` proxy DIAGNOSTIC,
not a corrected principal-product benchmark or multistep stock planning.
Comparisons are descriptive under unequal recorded work/completion, not
matched-compute superiority. Every method uses the same source registry.

All17 original analysis files are byte-preserved; the complete metrics JSON
is losslessly gzipped. All44 supported reactions and exact paired run/target
memberships are in `case_registry.csv`. The source analysis manifest SHA is
`11e49dd8695ca3b1ed3cdf46e7dbad449f3b299fc8aa35591467cbb7999d8056`.
Raw primary documents remain outside Git. The [six-case RDKit gallery](../additional_preparation_gallery_20261019_v11/README.md)
and [stereo-qualified unresolved source review](../primary_stereo_review_20261019/README.md)
separate actual net precedent from an unverified complete identity.

Fresh whole-repository CPU software gate:1,253 passed/14 skipped, one existing
PEFT warning,188.16s. No model/GPU task ran. Wider primary review and final
whole-goal review are not claimed complete.
