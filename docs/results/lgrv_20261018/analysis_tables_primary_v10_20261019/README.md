# Complete v10 evidence analysis, unchanged frozen predictions

Two further performed preparations were checked in their original documents:
US7173043B2 Example27.4, with its referenced Example27.1 method, and US5110940
Example108. Independently transcribed named structures match the frozen
generated reactions. No individual yield is borrowed from another preparation;
the oxidation temperature is unstated, and the source's elemental-analysis
discrepancy remains documented. See the [two-case gallery](../additional_preparation_gallery_20261019_v10/README.md).

All18,210 unique reactions, six conditions, target populations, corpus,
stereo/local rules and cap5 remain unchanged. All21 sealed output hashes and
complete original query/context/hit sequences were checked. There are38 exact
source-passage-checked net reactions (TierA),18,172 unresolved (TierC), and no
TierB/D. This is source-review coverage, not population chemical accuracy or
experimental validation of each symbolic FLOW arrow.

| Frozen condition | Original targets | Observed reports | Supported reactions across all terminal contexts | Targets with such a reaction |
|---|---:|---:|---:|---:|
| R1 MCTS |460|200|16|15|
| R1 Greedy I |460|460|5|5|
| R1 Greedy II |460|460|5|5|
| Single-reaction MCTS,10 total attempts |3,120|3,120|17|17|
| Single-reaction Greedy I |3,120|3,120|17|17|
| Single-reaction Greedy II |3,120|3,120|16|16|

The same reaction can occur in several rows, so rows cannot be summed to38.
R1 MCTS retains199 nonzero frozen-stock AND closures,10 targets with any
exact-supported step on a stock-closed route, and1 with every essential step
supported. Both Greedies retain zero supported-step stock-closed targets;
their5 supported reactions each in other contexts remain included. The260
pending MCTS targets remain pending; an already-stock target is not new success.

Single-reaction results remain the frozen `rxn_prod_min` proxy DIAGNOSTIC,
not a corrected principal-product benchmark or multistep stock planning.
Unequal work/completion allows descriptive comparison, not matched-compute
superiority. The same registry benefits both Greedies.

The17 whole analysis files are byte-preserved; only the complete metrics JSON
is losslessly gzipped. `case_registry.csv` includes all38 supported reactions
with exact paired run/target memberships, bound to the full analysis manifest
SHA `59adfab1ffd1e9e6554d66f7e85307efa89e3a96a29173e31739cd11f0ac10dd`.
Raw primary documents stay outside Git. The [complete review-state inventory](../primary_review_states_v10_20261019/README.md)
distinguishes inspected unresolved cases from cached but unreviewed documents.

No production code changed after the full CPU software gate of1,253 passed/
14 skipped/one existing PEFT warning,193.94s. The later90-test focused gate
passed44.46s. These are software checks, not experiments. No model/GPU job ran.
Wider original-source review and the final whole-goal audit remain incomplete.
