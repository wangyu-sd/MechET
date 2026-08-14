# FlowER-derived baseline evidence and fair-comparison view

Status: provisional evidence inventory, 2026-08-14. This document organizes
the results that already exist. It is not yet a cross-model leaderboard.

Machine-readable companion:
[`results/flower_baseline_evidence_inventory_20260814.json`](results/flower_baseline_evidence_inventory_20260814.json).

## Executive conclusion

The repository currently contains one complete result under the frozen MechET
protocol and several useful historical baseline results under related but
different FlowER-derived protocols. The historical results should be retained
as evidence, but a numerical ordering across rows would conflate:

- elementary-transition prediction with complete-reaction prediction;
- product-only input with a complete successor-state molecular world;
- full-world, main-fragment, and structural-precursor targets;
- beam-ranked Top-k with generation-order Pass@k;
- complete runs with failed or partial reproductions.

The fair view is therefore a set of protocol-matched blocks, followed by an
explicit experiment-completion matrix.

## Dataset lineage and denominator

The different test sizes are successive representations or filters of the
same FlowER-derived source, not interchangeable denominators.

```text
218,997 elementary transition rows
          | group by trajectory_id
          v
 28,971 complete reaction trajectories
          | compile with the current MECH_PROOF exporter
          v
  3,562 proof-compatible inverse trajectories
          | build Tool-SFT trace and replay in the executor
          v
  3,080 executable MechET inverse traces
```

There are 28,971 unique trajectory IDs in the 218,997 transition rows, with
7.559 transitions per trajectory on average. A transition-level accuracy
weights a long reaction multiple times; a reaction-level accuracy weights it
once.

## Frozen fair endpoint contracts to use for new experiments

The benchmark has two non-interchangeable tracks. The primary retrosynthesis
endpoint comparison uses all 28,971 official reaction-level test IDs. The
3,080 replay-compatible cases are a secondary executable-trace track.

| Axis | Frozen requirement |
|---|---|
| Unit | One complete reaction per stable test ID |
| Full endpoint test | All 28,971 official FlowER reaction/trajectory IDs; missing outputs are failures |
| Executable trace test | The same 3,080 frozen replay-compatible IDs; selection coverage is reported as 3,080/28,971 |
| Observable chemistry | Product molecular graph only; no product-side environment |
| Endpoint target | Unordered multiset of all atom-contributing structural precursor fragments |
| Normalization | Remove atom maps, RDKit canonicalize, fragment-order invariant, retain stereochemistry |
| Secondary endpoint | A separately labelled neutralized/canonical metric |
| Candidate count | At most 10 candidates per target |
| Ranking | Frozen, gold-independent rank; ranked Top-1/5/10 |
| Leakage | Zero train/test overlap under the frozen structural reaction key |

MechET may assign deterministic atom-map IDs internally because its tools need
atom addresses. The observable molecular graph must remain the same as the
unmapped product supplied to endpoint baselines; arbitrary map numbers must
not be treated as chemical evidence.

The complete endpoint export is built by
`scripts/build_flower_full_endpoint_sft.py` into
`data/flower_full_endpoint_sft/{train,valid,test}.jsonl`. It contains 257,171,
2,890, and 28,971 rows and performs no proof/compiler/replay filtering. It is
endpoint supervision, not executable-mechanism supervision.

## Block A: current frozen MechET result

This is the only completed result on the current 3,080-target executable-trace
test. It receives the product and constructs the precursor with environment-
owned tools. No textbook corpus is used.

| System | Unit | Target | Candidate semantics | @1 | @5 | @10 |
|---|---|---|---|---:|---:|---:|
| MechET Qwen3-8B, 3 epochs | Complete executable trajectory, n=3,080 | Structural precursor multiset | Generation-order Pass@k | 75.00% | 89.64% | 92.11% |
| Same run | Complete executable trajectory, n=3,080 | Mapped structural endpoint | Generation-order Pass@k | 65.78% | 84.35% | 87.99% |
| Same run | Complete executable trajectory, n=3,080 | Formal execution | Generation-order Pass@k | 94.03% | 99.64% | 99.74% |

The frozen formal-trace selector selects one generated candidate per target:

- structural exact: 82.89% (2,553/3,080);
- mapped exact: 74.09%;
- executable: 99.74%.

This selected result is a gold-independent reranking diagnostic, not NLL-
ranked Top-1. The candidate artifacts do not contain a frozen probability
rank, so Pass@k must not be relabelled as Top-k.

Evidence: [`results/flower_qwen3_8b_3epoch_k10.json`](results/flower_qwen3_8b_3epoch_k10.json).

## Block B: closest existing reaction-level product-only proxies

These runs share the broad `product -> precursors` direction and are the most
useful historical proxies. They are not yet head-to-head results because they
use 28,970/28,971 reactions and predict the complete reverse world. Their
`main-only` score checks only the largest precursor fragment, whereas MechET's
structural score requires every atom-contributing precursor fragment.

| System | n | Strict Top-1/5/10 | Main-only Top-1/5/10 | Valid decode | What remains unmatched |
|---|---:|---:|---:|---:|---|
| RXNGraphormer `orbit_main_product` | 28,971 | 5.30 / 9.28 / 10.54% | 51.34 / 61.04 / 62.99% | 53.58% per candidate | Different IDs and full-world target |
| Molecular Transformer, 100 epochs | 28,971 | 0.024 / 0.028 / 0.031% | 87.05 / 96.50 / 98.24% | 98.70% | Different IDs and full-world target |
| Graph2SMILES | 28,970 | 2.90 / 3.72 / 3.75% | 16.85 / 17.71 / 18.00% | 100% | One dropped row; different target |

The large Molecular Transformer gap between main-only and strict accuracy is
direct evidence that “largest precursor recovered” and “complete precursor
answer recovered” are different tasks. Neither column estimates MechET's
structural-multiset exact match.

Evidence paths:

- RXNGraphormer main-product: `/aaa/fionafyang/buddy1/whaleywang/reflow/outputs/paper_results/baseline_benchmark/rxngraphormer_main_product/test/metrics_main_only.json`
- Molecular Transformer: `/aaa/fionafyang/buddy1/whaleywang/reflow/outputs/paper_results/baseline_benchmark/molecular_transformer/test/metrics_100ep.json`
- Graph2SMILES: `/aaa/fionafyang/buddy1/whaleywang/reflow/outputs/paper_results/baseline_benchmark/g2s/test/metrics_full.json`

Allowed interpretation: these results establish functioning historical
product-only sequence baselines and expose the cost of full-world recovery.

Disallowed interpretation: MechET is better or worse than any row above based
on the displayed percentages.

## Block C: elementary-transition predecessor prediction

RXNGraphormer `orbit_retro` is a valid result for a different task. Each of
the 218,997 elementary transitions is evaluated independently:

```text
complete successor-state world -> complete predecessor-state world
```

| System | Unit | n | Strict Top-1/5/10 | Main-only Top-1/5/10 | Valid decode |
|---|---|---:|---:|---:|---:|
| RXNGraphormer `orbit_retro` | Elementary transition | 218,997 | 53.13 / 68.28 / 69.60% | 66.86 / 79.75 / 80.89% | 58.60% per candidate |

This measures one-step predecessor-state exact match, not trajectory exact
match, not product-only retrosynthesis, and not electron-transfer trajectory
accuracy. It is valuable as a learned local inverse-dynamics baseline. It
belongs in a transition-level table or an ablation, not the primary complete-
reaction endpoint table.

Evidence: `/aaa/fionafyang/buddy1/whaleywang/reflow/outputs/paper_results/baseline_benchmark/rxngraphormer_retro/test/metrics_main_only.json`.

## Block D: related tasks and provisional internal evidence

| Result | n | Existing observation | Appropriate use |
|---|---:|---|---|
| RXNGraphormer `orbit_completion` | 28,894 | Strict Top-1/10 = 0%; main-only Top-10 = 0.25% | Environment-completion evidence only |
| Old ORBIT v2 endpoint baseline | 500 endpoint cases | Top-1 exact = 62.4% | Internal pre-MechET baseline after full rerun |
| Old ORBIT v2 mechanistic block | 2,000 cases | RC F1 = 57.46%; BE trajectory MAE = 0.0160 | Mechanistic historical diagnostic |

The old ORBIT learned world model and RXNGraphormer are distinct systems.
Old ORBIT can become an internal baseline, but its current subset and legacy
target contract prevent direct placement in the primary table.

Evidence paths:

- RXNGraphormer completion: `/aaa/fionafyang/buddy1/whaleywang/reflow/outputs/paper_results/baseline_benchmark/rxngraphormer_completion/test/metrics_extended.json`
- Old ORBIT v2 endpoint: `/aaa/fionafyang/buddy1/whaleywang/reflow/outputs/paper_results/orbit_v2_benchmark/endpoint_baseline.json`

## Block E: artifacts that document failed or incomplete reproduction

These artifacts are useful engineering evidence but must not be interpreted as
the scientific performance of the named model.

| System | Nominal n | Symptom | Status |
|---|---:|---|---|
| LocalRetro | 28,971 | Only 4.56% non-empty outputs; all exact scores zero | Invalid/misaligned run |
| NeuralSym | 28,971 | Only 0.010% non-empty outputs; all exact scores zero | Invalid/misaligned run |
| Retroformer | 9,153 | Partial denominator; strict zero despite 99.46% valid decode | Incomplete or direction/adapter mismatch |
| FlowER inverse historical run | 69,279 output records | OOM batches replaced by placeholders; final evaluator raised `IndexError` | Failed inference; no score |

The zero scores above must not be cited as model-level baselines. GraphRetro,
ELECTRO, and DeepMech have code or planning artifacts but no completed matched
result in the inspected output tree.

## Provisional paper table

Until matched experiments are complete, the honest primary table has one
numeric row and explicit pending cells:

| System | Full 28,971 endpoint test | Same 3,080 trace IDs | Product-only | Structural target | Frozen rank | Endpoint result | Mechanism result |
|---|:---:|:---:|:---:|:---:|:---:|---:|---:|
| MechET Qwen3-8B | Pending | Yes | Yes | Yes | No: Pass@k only | 75.00 / 89.64 / 92.11 Pass@1/5/10 on trace subset | Execute Pass@1/5/10 = 94.03 / 99.64 / 99.74 |
| RXNGraphormer main-product | Historical unmatched | No | Yes | No | Yes | Pending matched rerun | N/A by architecture |
| Molecular Transformer | Historical unmatched | No | Yes | No | Yes | Pending matched rerun | N/A by architecture |
| Graph2SMILES | Historical unmatched | No | Yes | No | Yes | Pending matched rerun | N/A by architecture |
| FlowER inverse | No valid run | No valid run | To be frozen | To be frozen | To be frozen | Pending | Pending |
| Old ORBIT/Reflow world model | No | No | Protocol uncertain | Protocol uncertain | No matched artifact | Pending | Pending |
| ELECTRO | No | No | No matched run | No matched run | No | Pending | Pending |

“N/A by architecture” is preferable to zero for trajectory metrics that an
endpoint-only model does not produce.

## Experiment-completion order

| Priority | Experiment | Why it comes next | Completion evidence |
|---:|---|---|---|
| 1 | Freeze the complete 28,971-ID endpoint export and structural targets | Establishes the unbiased full endpoint denominator | Manifest, hashes, zero duplicate IDs, 100% source coverage |
| 2 | Train/evaluate RXNGraphormer on product -> structural precursors | Strongest existing learned graph-sequence baseline | Ranked Top-1/5/10 and valid decode |
| 3 | Evaluate a frozen MechET ranker or preserve Pass@k as a separate column | Resolves beam Top-k versus sampled Pass@k | Gold-independent score saved per candidate |

### Prepared full-endpoint evaluation

The full-endpoint Qwen3-8B checkpoint is evaluated with
`scripts/run_taiji_flower_full_endpoint_test_k10.sh`. The launcher freezes two
non-interchangeable reports from the same candidate artifact:

- all 28,971 official FlowER reaction-level test IDs;
- the 3,080 IDs matching the executable-trace test.

Ten independent endpoint candidates are retained per target. Generation order
is reported as Pass@1/3/5/10. Every candidate is then teacher-forced under the
same frozen adapter, ranked by assistant-token mean NLL without gold labels,
and reported separately as ranked Top-1/3/5/10. The matched subset is built by
`scripts/build_flower_endpoint_matched_subset.py`; its manifest records the one
legacy structural-reference mismatch and the one mapped-target mismatch rather
than silently deleting either row.
| 4 | Train/evaluate Molecular Transformer and Graph2SMILES on the same export | Covers canonical sequence and graph-to-sequence families | Same-ID ranked metrics |
| 5 | Repair and rerun FlowER inverse at reaction/trajectory level | Supplies the closest mechanistic baseline | Endpoint plus step/trajectory metrics, no placeholders |
| 6 | Adapt old ORBIT/Reflow world model | Quantifies gain over the prior internal method | Same IDs, target, normalization, denominator |
| 7 | Add GraphRetro/LocalRetro and ELECTRO/DeepMech | Expands template/semi-template and mechanism families | Valid, non-empty, protocol-matched runs |

Every new baseline row should record model revision, training split hash,
test-ID hash, target hash, prediction hash, candidate count, ranking policy,
normalization version, missing-output count, and overlap-audit result.

## Claims supported now

Supported:

- MechET produces executable inverse traces on the frozen 3,080-case subset.
- Its structural endpoint coverage is 75.00/89.64/92.11 Pass@1/5/10.
- RXNGraphormer is a functioning strong local predecessor-state predictor on
  218,997 FlowER transitions.
- Existing product-only baselines show a substantial distinction between
  recovering the largest precursor and recovering the complete reaction world.

Not yet supported:

- MechET outperforms RXNGraphormer, Molecular Transformer, or FlowER under a
  matched complete-reaction endpoint protocol.
- Pass@10 is equivalent to beam Top-10.
- Failed LocalRetro/NeuralSym/Retroformer reproductions reflect the intrinsic
  capability of those models.
- The 3,080-case proof-compatible subset represents all 28,971 FlowER test
  reactions without selection effects.
