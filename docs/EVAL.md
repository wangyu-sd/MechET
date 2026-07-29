# MechET evaluation

MechET = **MECH_ET v3** (mechanism-graph CoT + `BE_DELTA` + precursor answer).

## Benchmark inventory

**Retrosynthesis & planning results (usable + rerunnable):** **[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)**

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_mechet_eval.py` | **One-shot** infer → eval → TSV |
| `scripts/audit_mechet_gold.py` | Gold data QC (~100% on valid; not model eval) |
| `scripts/infer_mechet.py` | Qwen (+ LoRA) → `generations.jsonl` + manifest |
| `scripts/eval_mechet_generations.py` | Model vs `metadata.initial_reactants` |
| `scripts/collect_mechet_results.py` | Export TSV aligned with completion strict columns |

## Metrics (match `flower_completion` baselines)

| Column | Meaning |
|--------|---------|
| `top1_strict` / `top5_strict` / `top10_strict` | RDKit canonical multiset match vs gold reactants |
| `top1_main_only` | Largest heavy-atom fragment match (G2S auxiliary metric) |
| `valid_precursors` | All answer fragments parse |
| `format_ok` → `be_delta_exact` → `electron_conserved` | Mechanism process funnel |
| `state_agree` | `<answer>` matches graph `PRECURSOR_STATE` |

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
# Gold QC
python scripts/audit_mechet_gold.py --data data/mechet_sft/valid.jsonl --limit 500

# Full model eval (greedy)
export QWEN_MODEL_PATH=/path/to/Qwen3-8B
python scripts/run_mechet_eval.py \
  --data data/mechet_sft/valid.jsonl \
  --adapter outputs/mechet_sft/adapter \
  --out-dir outputs/mechet_eval/valid_run

# Beam top-5 (set num-return-sequences <= num-beams)
python scripts/run_mechet_eval.py \
  --data data/mechet_sft/test.jsonl \
  --adapter outputs/mechet_sft/adapter \
  --num-beams 5 --num-return-sequences 5 \
  --out-dir outputs/mechet_eval/test_beam5
```

Predictions JSONL fields: `prediction`, `candidates` (parsed answers), `raw_generations` (full texts), `gold_answer`.

Use **valid/test** for paper tables; `overfit32` is smoke only.
