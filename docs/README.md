# MechET documentation map

This page distinguishes current method specifications from historical files and defines the reading order for collaborators.

## Reading order for collaborators

1. [`../README.md`](../README.md) — scientific story, implemented system and runnable entrypoints.
2. [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) — authoritative ICLR scientific and execution contract.
3. [`MECHANISTIC_PRIMITIVE_LIBRARY.md`](MECHANISTIC_PRIMITIVE_LIBRARY.md) — Web evidence, primitive extraction, executable schema and model enhancement.
4. [`PROOF_CARRYING.md`](PROOF_CARRYING.md) — proof language and deterministic executor.
5. [`FORWARD_ELECTRON_EXPERT.md`](FORWARD_ELECTRON_EXPERT.md) — compact forward expert.
6. [`FRAMEWORK_MIGRATION.md`](FRAMEWORK_MIGRATION.md) — TRL, scale backends and Syntheseus planning.

## Authoritative documents

| Document | Purpose |
|---|---|
| [`../README.md`](../README.md) | Public architecture, current status, quickstart and method boundaries |
| [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) | Only source of truth for headline claims, required baselines, result tables, stopping gates and collaborator work packages |
| [`MECHANISTIC_PRIMITIVE_LIBRARY.md`](MECHANISTIC_PRIMITIVE_LIBRARY.md) | Source/license registry, evidence extraction, primitive schema, retrieval, model augmentation and matched ablations |
| [`PROOF_CARRYING.md`](PROOF_CARRYING.md) | `MECH_PROOF v1`, deterministic executor and proof semantics |
| [`FORWARD_ELECTRON_EXPERT.md`](FORWARD_ELECTRON_EXPERT.md) | Forward data, model, training, generation, calibration and integration |
| [`FRAMEWORK_MIGRATION.md`](FRAMEWORK_MIGRATION.md) | Framework-neutral environment, TRL reference path, scale backends and planning migration |

## Current companion protocols

| Document | Purpose |
|---|---|
| [`PROOF_EQUIVALENCE.md`](PROOF_EQUIVALENCE.md) | Partial-order equivalence, proof-class deduplication, MechComp-OOD and failure certificates |
| [`DATA_LEAKAGE_AND_ICLR_PLAN.md`](DATA_LEAKAGE_AND_ICLR_PLAN.md) | Dataset lineage, benchmark freezing, overlap audit and decontamination |
| [`../knowledge/README.md`](../knowledge/README.md) | Mechanism source registry, download gates, raw evidence and primitive asset policy |
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
2. Learned forward/selectivity scores and primitive-reference matches are soft evidence and must not be described as experimental truth.
3. A Web source or LLM extraction becomes a released primitive only after provenance capture, executable encoding and chemistry review.
4. Agent-training and planning frameworks must wrap `MechETAgentEnv`; they must not duplicate chemistry state or reward definitions.
5. Download revisions, hashes, licenses, quarantine records, configs and calibration thresholds must accompany reported checkpoints.
6. Restricted sources require explicit license acceptance; no downloader may bypass upstream access terms.
7. The README must not report unreleased checkpoints or non-frozen numbers as validated performance.
8. Historical documents must retain clear deprecated/archived markers and links to replacements.
