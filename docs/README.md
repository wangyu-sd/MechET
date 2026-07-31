# MechET documentation map

This page distinguishes current method specifications from historical files.

## Authoritative documents

| Document | Purpose |
|---|---|
| [`../README.md`](../README.md) | Project architecture, quickstart and complete runnable entrypoints |
| [`PROOF_CARRYING.md`](PROOF_CARRYING.md) | `MECH_PROOF v1`, deterministic executor and proof semantics |
| [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) | Authoritative proof-centric data, training, inference and paper evaluation contract |
| [`FORWARD_ELECTRON_EXPERT.md`](FORWARD_ELECTRON_EXPERT.md) | Compact forward expert: data acquisition, standardization, model, training, inference, generation, evaluation and integration |

## Current companion protocols

| Document | Purpose |
|---|---|
| [`PROOF_EQUIVALENCE.md`](PROOF_EQUIVALENCE.md) | Partial-order proof equivalence, hypothesis deduplication, MechComp-OOD and failure certificates |
| [`DATA_LEAKAGE_AND_ICLR_PLAN.md`](DATA_LEAKAGE_AND_ICLR_PLAN.md) | Dataset lineage, benchmark freezing, overlap audit and decontamination |
| [`../data/README.md`](../data/README.md) | Existing proof dataset construction |
| [`../data/FORWARD_EXPERT.md`](../data/FORWARD_EXPERT.md) | Forward-expert local data and checkpoint layout |

## Archived or deprecated documents

These remain only for provenance and must not define the current method.

| Document | Status | Replacement |
|---|---|---|
| [`EXPERIMENT_PLAN_ICLR_TO_NMI.md`](EXPERIMENT_PLAN_ICLR_TO_NMI.md) | Deprecated experiment plan | [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) |
| [`EVAL.md`](EVAL.md) | Deprecated `MECH_ET v3` guide | README and current method documents above |
| [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) | Historical inventory; not a result table | frozen manifests and authoritative experiment plans |
| [`README_DESIGN_NOTES.md`](README_DESIGN_NOTES.md) | Archived internal design notes | current README |

## Documentation policy

1. The deterministic executor remains the hard source of formal validity.
2. Learned forward/selectivity scores are soft, calibrated evidence and must not be described as experimental truth.
3. Download revisions, manifests, quarantine records, split policy, configs and calibration thresholds must accompany reported checkpoints.
4. Restricted datasets require explicit license acceptance; no downloader may bypass upstream access terms.
5. The README must not report unreleased checkpoints or non-frozen numbers as validated performance.
6. Historical documents must retain clear deprecated/archived markers and links to replacements.
