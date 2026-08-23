# Benchmark results and historical inventory

> [!CAUTION]
> **INCOMPLETE TRACE-VIEW RESULT — NOT A FULL BENCHMARK.** Every `3,080`
> FlowER result in this file covers only the replay-compatible subset of the
> complete `28,971`-reaction test set. It is diagnostic only and is prohibited
> from headline/full-corpus comparisons.

## Completed incomplete trace-view diagnostics (not headline results)

### FlowER inverse trace, Qwen3-8B, three-epoch Tool-SFT — incomplete 3,080/28,971 view

This is a completed same-source FlowER test run, not an external-transfer
result. The model receives only the mapped product and must construct the
precursor through environment-owned electron-flow tools. It does not use the
textbook corpus. The incomplete trace-view subset contains 3,080 targets; ten independent sampled
rollouts were retained for every target (30,800 candidates total).

| Metric | Pass@1 | Pass@5 | Pass@10 |
|---|---:|---:|---:|
| Structural endpoint | 75.00% | 89.64% | 92.11% |
| Mapped endpoint | 65.78% | 84.35% | 87.99% |
| Formal execution | 94.03% | 99.64% | 99.74% |
| Trace bound | 94.03% | 99.64% | 99.74% |

The frozen gold-independent formal-trace selector chooses one candidate per
target. Its selected structural exact rate is 82.89% (2,553/3,080), mapped
exact rate is 74.09%, and execution rate is 99.74%. These selector diagnostics
are separate from generation-order Pass@K.

Candidate rollouts have no frozen probability rank. Therefore the table is
reported as generation-order **Pass@K**, not ranked Top-K or beam-search
accuracy. Missing predictions remain failures in the denominator.

Run identity:

```text
task: mechet_flower_qwen3_8b_3ep_test_k10_8a100_20260813_01
base model: Qwen/Qwen3-8B
base revision: b968826d9c46dd6066d109eabc6255188de91218
adapter SHA-256: 2970d8ab5ad8e10947f1b484a1dbe3ab6ca226ffaa6c8dbf7bf529966a4c2c8c
reference SHA-256: 047e8bcb9b0abc57e44adad2f6406d69f1521d4d86fba21cc9a550d9c8b0b57b
predictions SHA-256: fa13349c61ab251f03fdf15aa098687e42c18947d647d24373094f5c968b2be8
stable-ID SHA-256: fe30eec946fab075da0ce63ed41034578735e20454d25031b80e4a792c36946b
runtime-contract SHA-256: 17e56b7b3a82de597bafee96b45e05035236ee656f21c4873cc042d8bb2904e7
seed: 17
temperature: 0.7
top-p: 0.95
max tool iterations: 24
```

The machine-readable summary is frozen in
[`results/flower_qwen3_8b_3epoch_k10.json`](results/flower_qwen3_8b_3epoch_k10.json).

## Historical inventory

> [!WARNING]
> The remainder of this file describes earlier internal runs. It contains
> non-matching tasks, incomplete evaluations, and results that were explicitly
> marked rerunnable or unsuitable for citation. Do not mix those entries with
> the frozen run above.

Use:

- [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) for the required current result tables, metrics, datasets, and stopping gates;
- [`DATA_LEAKAGE_AND_ICLR_PLAN.md`](DATA_LEAKAGE_AND_ICLR_PLAN.md) for benchmark freezing and decontamination;
- frozen run manifests under `outputs/` for actual completed experiments.

The former inventory, including old baseline numbers and internal filesystem
paths, remains available in Git history. No number from that inventory should
enter a paper table without being rerun on the frozen current protocol.
