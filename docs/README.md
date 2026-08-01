# MechET documentation map

This page distinguishes current method specifications from historical files and defines the reading order for collaborators.

## Recommended reading order

1. [`../README.md`](../README.md) — scientific story, current architecture, runnable entrypoints, and project boundaries.
2. [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) — authoritative ICLR collaboration contract, work packages, experiments, metrics, stopping rules, and required paper package.
3. [`PROOF_CARRYING.md`](PROOF_CARRYING.md) — executable proof language and deterministic executor semantics.
4. [`FORWARD_ELECTRON_EXPERT.md`](FORWARD_ELECTRON_EXPERT.md) — compact forward expert data, model, training, inference, generation, and calibration.
5. [`FRAMEWORK_MIGRATION.md`](FRAMEWORK_MIGRATION.md) — framework-neutral environment, TRL reference path, scale backends, and Syntheseus planning.

## Authoritative documents

| Document | Purpose |
|---|---|
| [`../README.md`](../README.md) | Project thesis, architecture, current implementation, quickstart, and end-to-end entrypoints |
| [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) | Authoritative ICLR scientific and execution contract: claims, data, models, RL, forward falsification, single-step and multistep evaluation, ownership, stopping rules, and artifact requirements |
| [`PROOF_CARRYING.md`](PROOF_CARRYING.md) | `MECH_PROOF v1`, deterministic executor, local operations, proof equivalence boundary, and formal failure semantics |
| [`FORWARD_ELECTRON_EXPERT.md`](FORWARD_ELECTRON_EXPERT.md) | Compact forward expert: data acquisition, standardization, model, training, inference, generation, selectivity, calibration, and integration |
| [`FRAMEWORK_MIGRATION.md`](FRAMEWORK_MIGRATION.md) | Framework-neutral `MechETAgentEnv`, TRL reference training, scale backends, Syntheseus planning, and migration stages |

## Current companion protocols

| Document | Purpose |
|---|---|
| [`PROOF_EQUIVALENCE.md`](PROOF_EQUIVALENCE.md) | Partial-order proof equivalence, hypothesis deduplication, MechComp-OOD, and failure certificates |
| [`DATA_LEAKAGE_AND_ICLR_PLAN.md`](DATA_LEAKAGE_AND_ICLR_PLAN.md) | Dataset lineage, benchmark freezing, overlap audit, and decontamination |
| [`../data/README.md`](../data/README.md) | Existing proof dataset construction and local layout |
| [`../data/FORWARD_EXPERT.md`](../data/FORWARD_EXPERT.md) | Forward-expert local data, manifests, and checkpoint layout |

## Archived or deprecated documents

These remain only for provenance and must not define the current method.

| Document | Status | Replacement |
|---|---|---|
| [`EXPERIMENT_PLAN_ICLR_TO_NMI.md`](EXPERIMENT_PLAN_ICLR_TO_NMI.md) | Deprecated experiment plan | [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) |
| [`EVAL.md`](EVAL.md) | Deprecated `MECH_ET v3` guide | README and the current method documents above |
| [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) | Historical inventory; not a result table | frozen manifests and the authoritative experiment plan |
| [`README_DESIGN_NOTES.md`](README_DESIGN_NOTES.md) | Archived internal design notes | current README |

## Documentation policy

1. The deterministic executor remains the hard source of formal validity.
2. Learned forward/selectivity scores are soft, calibrated evidence and must not be described as experimental truth.
3. Agent-training and planning frameworks must wrap `MechETAgentEnv`; they must not duplicate chemistry state or reward definitions.
4. Download revisions, manifests, quarantine records, split policy, configs, and calibration thresholds must accompany reported checkpoints.
5. Restricted datasets require explicit license acceptance; no downloader may bypass upstream access terms.
6. The README must not report unreleased checkpoints or non-frozen numbers as validated performance.
7. Historical documents must retain clear deprecated/archived markers and links to replacements.
8. The ICLR experiment plan is the only source of truth for headline claims, required baselines, result tables, stopping gates, and collaborator work packages.
