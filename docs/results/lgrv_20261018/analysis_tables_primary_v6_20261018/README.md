# Complete five-registry source review, v6

All18,210 frozen unique generated reactions and all six original conditions
were reaggregated. The original target universes, corpus, full-stereo retrieval,
radius and cap are unchanged. The entire query, retrieval-context and210-hit
row sequences were compared with v5 and are identical. Only six additional
performed primary preparations were added.

Result: **15 TierA;18,195 TierC;0 TierB/D**. This is source-supported coverage,
not population chemical accuracy. The21 original output hashes were verified;
analysis manifest SHA256:
`e633a40221482671c180d66d864bb36db4a5d42809c10602e00bd942b08aaa41`.

| Frozen condition | Original targets | Observed / pending | Current source-supported outcome |
|---|---:|---:|---|
| R1 MCTS |460|200 /260|199 nonzero stock-AND closures;7 with an exact step on a stock-closed route;1 with every essential step A/B |
| R1 Greedy I |460|460 /0|38 stock closures;0 source-covered stock routes |
| R1 Greedy II |460|460 /0|36 stock closures;0 source-covered stock routes |
|31k MCTS |3,120|3,120 /0|4 targets with a source-checked complete reaction |
|31k Greedy I |3,120|3,120 /0|3 targets with a source-checked complete reaction |
|31k Greedy II |3,120|3,120 /0|2 targets with a source-checked complete reaction |

Across all generated R1 terminal contexts, MCTS has10 distinct checked
reactions affecting9 targets. Greedy I/II have2/3 checked reactions affecting
2/3 targets. These include partial/unclosed route contexts and are **not**
counts of fully source-covered routes. V5's zero checked Greedy reactions is
not carried into v6: the same new source registry was applied to every method.

The31k data remain the frozen `rxn_prod_min` proxy **diagnostic**, not a
corrected principal-product benchmark or a stock-route task. R1 work is unequal;
comparisons remain descriptive, with260 MCTS reports still pending.

## Six newly checked preparations

| Performed net transformation | Primary locator | Reported product |
|---|---|---|
| Aminochlorodihydrobenzofurancarboxylic acid → acid chloride |[US5872131](https://patents.google.com/patent/US5872131/en),Example2(a) |4.8g,100% **crude** residue; not a purified yield |
| N-Boc-3-hydroxyazetidine → O-mesylate |[US6660744B1](https://patents.google.com/patent/US6660744B1/en),Example110(b) |4.85g/0.019mol oil; NMR; no percentage inferred |
|5-Cyano-2-methylaniline + chloroacetyl chloride → chloroacetanilide |[US6969713B2](https://patents.google.com/patent/US6969713B2/en),Example92(i) |1.85g white solid; MS; not the following chiral coupling |
|4-Amino-3-methylbenzamide + thiophosgene → aryl isothiocyanate |[US8785460B2](https://patents.google.com/patent/US8785460B2/en),Reference198 |91%; NMR/MS |
| Cyano/nitrobenzenesulfonyl chloride + methylamine → sulfonamide |[US8252795B2](https://patents.google.com/patent/US8252795B2/en),Intermediate71C |31%; chromatography/NMR |
|3-Chlorobenzoyl chloride +4-aminopyridine → amide |[US8975288B2](https://patents.google.com/patent/US8975288B2/en),compound11 |52%; chromatography/NMR |

The [new registry](../../../../configs/analysis/lgrv_primary_sources_20261018_v5.json)
records independently transcribed named structures, conditions, original/raw
document hashes, short quote hashes, exact offsets and limitations. All six
matched unchanged frozen reaction keys. Assistant source checks are not a
staffed human review or physical verification of each symbolic electron arrow.

`case_registry.csv` contains **all15** currently supported reaction keys and
every observed method/target membership. `metrics.json.gz` is the lossless whole
original JSON; all17 distributed analysis artifacts are complete files, not
selected successful rows. The distribution manifest retains original Ceph
paths/hashes for omitted large bindings.

The [coverage audit](../coverage_traceback_20261018/README.md) explains the low
initial source support and recovers alternate patent pointers. Existing Figure2/3
exports are still explicitly **v5**, not relabelled with these newer numbers.
The [complete v6 review queue](../coverage_review_v6_20261018/README.md) retains
all152 currently unresolved candidate queries, not only the earlier v5 queue.
Broader primary/analogue review, remaining controls and final whole-goal review
remain unfinished; this is an incremental delivery, not completion of PR89.
