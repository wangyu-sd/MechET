# Complete six-registry primary-source results, v7

The same18,210 frozen unique generated reactions and all six original
conditions were reaggregated with seven additional directly inspected
preparations. The complete query, retrieval-context and210-hit row sequences
were checked identical to v6. Original source indices, radius, cap, actor,
target populations and all earlier results are unchanged.

Result: **22 TierA;18,188 TierC;0 B/D**. All21 output hashes were checked;
analysis manifest SHA256:
`738494846ceb0e4fbb2213c67e5c1fb399558be87c5c26b02c5c9eadc25f9311`.
These are source-coverage counts, not a population chemical-accuracy estimate.

| Frozen condition | Original targets | Observed / pending | Source-supported outcome |
|---|---:|---:|---|
| R1 MCTS |460|200 /260|199 nonzero stock-AND closures;8 targets with an exact step on a closed route;1 target with all essential steps A/B |
| R1 Greedy I |460|460 /0|38 stock closures;0 targets with all-step-covered stock routes |
| R1 Greedy II |460|460 /0|36 stock closures;0 targets with all-step-covered stock routes |
|31k MCTS |3,120|3,120 /0|7 targets with a checked complete reaction |
|31k Greedy I |3,120|3,120 /0|5 targets with a checked complete reaction |
|31k Greedy II |3,120|3,120 /0|4 targets with a checked complete reaction |

Across all R1 terminal contexts, MCTS has14 distinct supported reactions in13
targets; Greedy I/II have3/4 in3/4 targets. Partial/unclosed contexts are included
in those counts, not relabelled as fully source-covered routes. The31k source
is still the frozen `rxn_prod_min` proxy **diagnostic**, not a corrected
principal-product or stock-route benchmark. Unequal work/completion remains
explicit; this is not a matched-compute causal comparison.

`case_registry.csv` includes all22 supported reaction keys and every observed
method/target membership, with its own manifest/hash. `metrics.json.gz` is the
lossless complete original JSON; all17 distributed analysis files are whole
files rather than selected positive rows. Original paths/hashes remain in the
distribution manifest for large native/program bindings not duplicated here.

The [seven new source-checked reaction plates](../additional_selectivity_gallery_20261018_v7/README.md)
show real generated chemistry, not simulated superiority examples. A separate
[primary-source positional-isomer discrepancy](../primary_structure_review_20261018/README.md)
is retained as unresolved; an indexed exact hit alone did not earn TierA.
The [current full review queue](../coverage_review_v7_20261018/README.md)
contains145 unresolved candidate queries. They are not automatically invalid.

Whole CPU software gate:1,229 passed/14 skipped/one existing PEFT warning,
164.11s. Seven new binding tests protect source drift, quote/identity mismatch,
unreviewed claims and overwrite safety. They are not model-performance results.
No new model inference, training, Taiji job or main/paper merge was performed.
Broader primary/analogue review, remaining controls and final whole-goal review
are still required; PR89 is not complete.
