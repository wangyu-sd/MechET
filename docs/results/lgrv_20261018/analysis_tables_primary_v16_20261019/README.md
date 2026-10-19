# Complete v16 literature-evidence tables

All 18,210 frozen complete-reaction queries and six recorded conditions are
retained: **47 Tier A, 12 query-scoped Tier B, and 18,151 unresolved**. Only
two previously unresolved reactions gain analogue support; all 57 previously
supported evidence rows and all query/context/indexed-candidate sequences
remain unchanged. The 59 supported generated reactions are not 59 independent
performed experiments or a population chemical-accuracy estimate.

This increment checks 12 source/query pairs across five primary patents:
two compatible analogues, five unresolved substrate-applicability comparisons,
and five source-side preparation/formulation limits. Source products, indexed
preparations, and unmeasured generated substrates remain distinct. The source
judgements are explicitly authored inputs, not independent policy calibration.

All 21 sealed outputs and current producer/registry hashes are checked. The
17 whole analysis files are byte-identical or losslessly gzipped copies; the
[distribution manifest](stored_artifacts_manifest.json) records both hashes.
The [59-case registry](case_registry.csv) includes every supported reaction and
its exact paired run/target memberships, structures, locator, and document
hashes. No supported case is selected by method advantage.

| R1 recorded condition | Saved reports / original targets | Distinct A/B-supported reactions | Targets with A/B support | Nonzero stock-AND closures |
|---|---:|---:|---:|---:|
| MCTS | 200 / 460 | 27 | 25 | 199 |
| Greedy I | 460 / 460 | 7 | 6 | 38 |
| Greedy II | 460 / 460 | 7 | 6 | 36 |

Reaction/target support above covers all terminal contexts, not necessarily
reactions on a complete stock-closed route. For MCTS, 11 targets have an exact
supported step on such a route, and one has every essential step supported.
Both Greedies retain zero for those two outcomes. The 260 unsaved MCTS targets
remain pending, not failed searches. These R1 counts are unchanged from v15.

| 31k frozen proxy condition | Original targets | Recorded attempt budget | Targets with A/B support | Targets with exact support |
|---|---:|---|---:|---:|
| MCTS | 3,120 | 10 total trajectory attempts | 26 | 22 |
| Greedy I | 3,120 | One greedy episode | 18 | 18 |
| Greedy II | 3,120 | One greedy episode | 17 | 17 |

The new two analogue supports occur in this single-reaction diagnostic. These
are complete-reaction literature-support outcomes, not reference-endpoint
accuracy, NLL-ranked Top-K, or stock-route planning. Unequal recorded work makes
method comparisons descriptive, not matched-compute causal gains.

New source reviews:

- [US8536165 Intermediate34: three comparisons](../grouped_primary_review_batch4_20261019/README.md).
- [US6326359 Preparation42: two comparisons](../grouped_primary_review_batch5_20261019/README.md).
- [US6960598 Example34(b): two unresolved comparisons](../grouped_primary_review_batch6_20261019/README.md).
- [US7465795 / US5068340: five preparation/formulation limits](../source_preparation_limits_20261019/README.md).

The [whole coverage audit](../coverage_review_v16_20261019/README.md),
[108-query work-state inventory](../primary_review_states_v16_20261019/README.md),
[six-condition summary](../POPULATION_RESULTS_PRIMARY_V16_20261019.json), and
[population figures](../population_figures_primary_v16_20261019/README.md)
retain the full protocol. No model inference, GPU/Taiji job, retrieval-policy
change, or main/paper merge was performed. Broader source review and the final
whole-goal audit remain open.
