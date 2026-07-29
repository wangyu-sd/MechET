# MechET evaluation

MechET = **MECH_ET v3** (mechanism-graph CoT + `BE_DELTA` + precursor answer).

## Benchmark inventory

**Retrosynthesis & planning results (usable + rerunnable):** **[docs/BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)**

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/audit_mechet_gold.py` | Gold data QC (~100% on valid; not model eval) |
| `scripts/infer_mechet.py` | Qwen (+ LoRA) → `generations.jsonl` |
| `scripts/eval_mechet_generations.py` | Model vs `metadata.initial_reactants` |
| `scripts/collect_mechet_results.py` | Export TSV aligned with completion strict columns |

## Metrics (match completion baselines)

| Column | Meaning |
|--------|---------|
| `top1_strict` | RDKit canonical multiset match vs gold reactants |
| `top5_strict` / `top10_strict` | Ranked beam (when enabled) |
| `valid_precursors` | All answer fragments parse |

Mechanism-only (secondary table): `format_ok`, `reachability_ok`, `be_delta_exact`, `electron_conserved`, `state_agree`, `by_topology`.

Implementation: `src/mechet/metrics.py`

## Data splits (MechET SFT)

| Split | Rows |
|-------|------|
| train | 258,233 |
| valid | 2,900 |
| test | 29,118 |

Build: `scripts/build_mechet_sft.py` from `flower_new_dataset`. Align test IDs with `flower_completion` (28,971) before merging into the baseline table.

## Workflow

```bash
python scripts/audit_mechet_gold.py --data data/mechet_sft/valid.jsonl --limit 500
export QWEN_MODEL_PATH=/path/to/Qwen3-8B
python scripts/infer_mechet.py --data data/mechet_sft/valid.jsonl --adapter outputs/.../adapter
python scripts/eval_mechet_generations.py --predictions outputs/mechet_eval/generations.jsonl
python scripts/collect_mechet_results.py --summary outputs/mechet_eval/model_eval_summary.json
```

Use **valid/test** for paper tables; `overfit32` is smoke only.
