# Complete v9 evidence analysis, unchanged frozen predictions

Five additional original performed preparations were checked, with named
structures independently transcribed and referenced general methods read.
All18,210 unique reactions, six original conditions and fixed target
populations remain. The corpus, stereo/local matching rules and cap5 did not
change. All21 sealed output hashes and complete query/context/hit sequences
were verified against the previous stage.

There are36 exact source-passage-checked net reactions (TierA),18,174 unresolved
reactions (TierC), and no TierB/D. This is source-review coverage, not population
chemical accuracy or experimental validation of every symbolic FLOW arrow.

| Frozen condition | Original targets | Observed reports | Supported reactions across all terminal contexts | Targets with such a reaction |
|---|---:|---:|---:|---:|
| R1 MCTS |460|200|16|15|
| R1 Greedy I |460|460|5|5|
| R1 Greedy II |460|460|5|5|
| Single-reaction MCTS,10 total attempts |3,120|3,120|15|15|
| Single-reaction Greedy I |3,120|3,120|16|16|
| Single-reaction Greedy II |3,120|3,120|15|15|

The same reaction can occur in several rows, so rows must not be summed to36.
R1 MCTS retains199 nonzero frozen-stock AND closures,10 targets with any
exact-supported step on a stock-closed route, and1 target with every essential
step supported. Both Greedies retain zero supported-step stock-closed targets;
their5 supported reactions each in other contexts are not discarded. Pending
260 MCTS targets remain pending; the already-stock target is not new success.

Single-reaction results remain the frozen `rxn_prod_min` proxy DIAGNOSTIC, not
a corrected principal-product benchmark or multistep stock planning. Unequal
recorded work/completion permits descriptive comparison only, not a causal
MCTS or matched-compute advantage. The same registry improves both Greedies.

The17 whole original analysis files are byte-preserved; only the complete
metrics JSON is losslessly gzipped. All36 supported reactions and exact paired
run/target memberships appear in `case_registry.csv`, bound by its manifest to
the full source analysis SHA
`de55dacfa7e87fc8dbf138267c4be16574993c1562dc247ca2a3ec953af11f46`.
Full raw source files stay outside Git with hashes; no whole primary patent is
redistributed. A separate three-case analogue scope review changes no grade.

Latest whole CPU software gate:1,253 passed/14 skipped, one existing PEFT
warning,193.94s. No model/GPU/training task ran. Main population figures remain
explicitv8 exports; the new five-case gallery isv9. Wider source/analogue work
and final whole-goal audit are not claimed complete.
