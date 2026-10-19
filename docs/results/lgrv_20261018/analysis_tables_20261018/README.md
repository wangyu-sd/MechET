# Full frozen analysis tables, not a sampled evaluation

These are byte-bound outputs of `source_checked_20261018_v3`, covering every
18,210 frozen unique reaction query, all original460 R1 target IDs and all
3,120 IDs in each single-reaction condition. R1 MCTS has200 observed reports
and260 pending, not460 completed searches. The31k source remains the frozen
`rxn_prod_min` proxy diagnostic, not a corrected principal-product benchmark.

`reaction_evidence.jsonl.gz`, `literature_hits.jsonl.gz`,
`route_evidence.jsonl.gz`, `source_passages.jsonl.gz`,
`baseline_comparison.csv`, `evidence_coverage.json` and `analysis_manifest.json`
are unchanged copies of the real output files. `metrics.json.gz` contains
**every byte of the complete original metrics.json** after gzip decompression;
it includes all per-target outcomes, paired comparisons and composition data.
No target rows were summarized away. File/hash correspondence is in
`stored_artifacts_manifest.json`.

The full original analysis manifest also identifies large original model
program links, generated reaction occurrences and native/raw output pointers
on Ceph. Those files remain preserved there, not silently replaced by these
bounded Git copies. Complete patent text and the18GB index are not republished.

Current evidence: four source-checked exact net reactions; all other tiers C
are unresolved, not automatically chemically invalid. The amount of primary
review is insufficient for a population chemical-success claim. No NLL/Top-K
ranking was introduced and no new inference/training was launched.
