# Complete v15 literature-evidence tables

All18,210 frozen queries andsix original conditions remain:**47 TierA,
10 query-scoped TierB,18,153 unresolved**. These57 supported generated
reactions are not57 independent performed experiments or a population
chemical success rate. Three new analogue judgements are added;all54
previously supported evidence rows remain unchanged.

The two additional source-review batches cover all10 assigned query/source
pairs:3 compatible analogues and7 unresolved comparisons. Their reviewed
preparations are335f/282d inUS8618107B2 and133 inUS8569331B2. Each substrate
pair has its own recorded differences andlimitations;compatibility is not
inherited across a local family or a handle-swapped route to the same product.

All21 sealed output/current producer hashes andcomplete original query/
context/candidate sequences were checked. The17 distributed whole files
are byte-identical orlosslessly gzipped,including full metrics JSON. The
[stored manifest](stored_artifacts_manifest.json) maps original/stored hashes.
The [57-case registry](case_registry.csv) retains every supported reaction,
exact paired run/target membership,source locator,structures anddocument hashes.

R1 MCTS has27 supported reactions across25 targets in all terminal contexts;
Greedy I/II each retain7 reactions across6 targets. Stock-route support
remains11 targets with an exact-supported step and1 with every essential
step supported,with460 original targets/200 saved MCTS reports/260 pending.
The31k frozen proxy now has24 supported MCTS targets(22exact+2analogues),
versus18/17 Greedy targets. All3,120 IDs andthe10-total-attempt MCTS budget
are preserved. Unequal-work comparisons remain descriptive,not causal gains.

| R1 recorded condition | Saved reports/original targets | Distinct A/B-supported reactions | Targets with an A/B-supported reaction | Nonzero stock-AND closures |
|---|---:|---:|---:|---:|
| MCTS |200/460|27|25|199|
| Greedy I |460/460|7|6|38|
| Greedy II |460/460|7|6|36|

Supported reactions/targets in this table cover all terminal contexts;they
are not necessarily part of a complete stock-closed route. For MCTS,
stock-route any-exact/every-essential-step support remains11/1 targets;
both Greedies remain0/0. Unfinished MCTS targets are pending,not failures.

|31k frozen proxy condition|Original targets|Attempt budget|Targets with A/B support|Targets with exact support|
|---|---:|---|---:|---:|
|MCTS|3,120|10 total trajectory attempts|24|22|
|Greedy I|3,120|One greedy episode|18|18|
|Greedy II|3,120|One greedy episode|17|17|

These are literature-support outcomes for generated complete precursor sets,
not reference endpoint accuracy,NLL-rankedTopK or a stock-route benchmark.

[US8618107 review](../grouped_primary_review_batch2_20261019/README.md),
[US8569331 review](../grouped_primary_review_batch3_20261019/README.md),
[whole coverage](../coverage_review_v15_20261019/README.md),
[all110 work states](../primary_review_states_v15_20261019/README.md) and
[six-condition summary](../POPULATION_RESULTS_PRIMARY_V15_20261019.json)
are included. No new model/GPU task ran. Wider primary review andthe final
whole-goal audit remain open.
