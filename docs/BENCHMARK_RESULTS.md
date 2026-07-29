# Benchmark results inventory (retrosynthesis & planning)

Scope: **single-step retrosynthesis** and **retrosynthesis planning** only.  
Parent repo: `/aaa/fionafyang/buddy1/whaleywang/reflow` (artifacts on disk, not all in Git).

Unified completion eval protocol: `reflow/flower_orbit_benchmark/flower_baseline_protocol.md` — input = main product, target = full precursor multiset, metric = **top-k strict EM** (RDKit canonical).

---

## Task A — Single-step retrosynthesis (`flower_completion`)

**Test set:** 28,971 · **Data:** `datasets/retro/data/flower_completion/test.txt`

### ✅ Usable (same task, completed eval)

| Model | top1 strict | top5 strict | top10 strict | top1 main-only | valid decode | Artifact (reflow) |
|-------|-------------|-------------|--------------|----------------|--------------|-------------------|
| **Graph2SMILES** | **2.90%** | 3.60% | **3.75%** | 16.85% | 100% | `outputs/paper_results/baseline_benchmark/g2s/test/` |
| **RxnGraphormer (main_product)** | **5.30%** | 9.28% | **10.54%** | 51.3% | 53.6% | `outputs/paper_results/baseline_benchmark/rxngraphormer_main_product/test/` |
| **Molecular Transformer (BART, 100 ep)** | **~0.02%** | ~0.03% | ~0.03% | 87.0% | 98.7% | `outputs/paper_results/baseline_benchmark/molecular_transformer/test/metrics_100ep.json` |

Notes:
- G2S trained 5k / 500k planned steps — conservative lower bound; PROVENANCE documents full pipeline.
- RxnGraphormer main_product = src single main product → full reaction world (correct completion task).
- Mol. Transformer: strict ≈ 0 (predicts main fragment only); useful as strict baseline, not main-only headline.

**MechET (ours):** no trained-model top-k yet. Gold QC only (`scripts/audit_mechet_gold.py`).  
MechET SFT test = **29,118** rows from `flower_new_dataset` — align with 28,971 before table merge.

### 🔁 Ran wrong / incomplete — **rerunnable** (do not cite as-is)

| Model | Issue | Rerun action | Artifact |
|-------|-------|--------------|----------|
| **FlowER-Retro** | 28.9% on **218,997** `flower_retro` steps, not completion 28,971; different metric pipeline | Re-infer on `flower_completion` test + `eval_baselines.py` strict EM | ckpt: `FlowER/checkpoints/orbit_retro/model.1110000_36.pt`; ref: `outputs/paper_results/all_metrics_20260507-2248/complexity/complexity_report.json` |
| **FlowER (forward)** | Same family as above; forward mechanistic, not completion retro | Optional supplementary only | `outputs/paper_results/baseline_benchmark/flower/` (predictions mostly empty — needs rerun) |
| **Retroformer** | 512-token limit → 31.7% train/test; strict **0%** on full completion | Retrain with longer context or report filtered-n=9153 with footnote | `outputs/paper_results/baseline_benchmark/retroformer/test/` |
| **RxnGraphormer (retro)** | Input includes full environment; **mechanistic step** task, not endpoint | Keep for supplementary mechanistic table only | `outputs/paper_results/baseline_benchmark/rxngraphormer_retro/test/` |
| **LocalRetro** | Train/test distribution shift; PROVENANCE: **DO NOT USE** | Retrain on completion train or mark N/A | `outputs/paper_results/baseline_benchmark/localretro/test/` |
| **NeuralSym** | Template extraction 0% on completion; PROVENANCE: **DO NOT USE** | Same as LocalRetro | `outputs/paper_results/baseline_benchmark/neuralsym/test/` |
| **Reflow GNN (Ours)** | Different paradigm (GNN world model), not LLM CoT; top1 **62.4%** on ~29k test | Reference upper bound only, separate table | `outputs/paper_results/all_metrics_20260507-2248/main_benchmark_test/` |

### Recommended MechET comparison columns

**Main table (vs rows above):** `top1_strict`, `top5_strict`, `top10_strict`, `valid_precursors`  
**Mechanism table (MechET only):** `format_ok` → `reachability_ok` → `be_delta_exact` → `electron_conserved` → `state_agree`, split by topology

Scripts: `scripts/infer_mechet.py` → `scripts/eval_mechet_generations.py` → `scripts/collect_mechet_results.py`

---

## Task B — Retrosynthesis planning (PaRoutes)

**Benchmark:** PaRoutes 1.0.0 n1 / n5 (EVAL_ONLY, 20,000 targets)  
**Protocol:** `reflow/data/orbit_benchmarks/eval_protocols/paroutes_full.json`  
**Primary metric:** route-level `validated_solved` under matched budget

### ✅ Usable

| Run | Status |
|-----|--------|
| *(none formal)* | No matched-budget n1/n5 result on record |

### 🔁 Ran wrong / incomplete — rerunnable

| Run | Result | Issue | Rerun |
|-----|--------|-------|-------|
| PaRoutes smoke (20 targets) | 0/20 solved | Integration smoke, not formal | After trained proposer: `scripts/run_orbit_benchmarks.py` + `paroutes_full.json` |
| G01 exec_mvp_20 | operator / full_long_horizon **7/20** | 20 diagnostic tasks, not PaRoutes protocol | `outputs/orbit_results_recovery/results/paroutes_dev_main_table.tsv` |

Registry: `reflow/docs/orbit_claim_evidence_registry.tsv` — `R2_PAROUTES_FORMAL = NOT_EXECUTED`

---

## Excluded (not listed)

Gold-data audits (~100%), overfit32 smoke, ORBIT-Hermes agent val50, EvoRetro R3b agent arms, LongPlan pilots, incomplete ORBIT-Qwen v0.1/v0.2 SFT rows — wrong task or inconsistent protocol.

---

## Quick reference paths (reflow)

```text
outputs/paper_results/baseline_benchmark/     # migrated baseline metrics + PROVENANCE.md
flower_orbit_benchmark/                       # unified 9-model protocol + converter
data/orbit_benchmarks/eval_protocols/         # PaRoutes formal protocol
reflow/data/orbit_mech_et_sft/                  # MechET training data (same lineage as MechET build)
```
