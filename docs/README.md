# MechET documentation map

This page defines which documents are authoritative for the current MechET-GFR project and which files are retained only for historical provenance.

## Authoritative documents

| Document | Purpose |
|---|---|
| [`../README.md`](../README.md) | Project overview, current capabilities, quickstart, and release status |
| [`PROOF_CARRYING.md`](PROOF_CARRYING.md) | `MECH_PROOF v1` language, electron-flow semantics, executor, and the distinction between local primitives and reaction templates |
| [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) | End-to-end experimental contract: data, models, losses, algorithms, inference, validation, required results, and stopping gates |

## Current companion protocols

| Document | Purpose |
|---|---|
| [`PROOF_EQUIVALENCE.md`](PROOF_EQUIVALENCE.md) | Partial-order proof equivalence, hypothesis deduplication, MechComp-OOD, and failure certificates |
| [`DATA_LEAKAGE_AND_ICLR_PLAN.md`](DATA_LEAKAGE_AND_ICLR_PLAN.md) | Dataset-lineage audit, FlowER–USPTO overlap controls, benchmark freezing, and decontamination policy |
| [`../data/README.md`](../data/README.md) | Dataset construction and local data layout |

## Archived or deprecated documents

These files remain in the repository so earlier experiments and discussions can be interpreted, but they must not be used as the current method or experiment specification.

| Document | Status | Replacement |
|---|---|---|
| [`EXPERIMENT_PLAN_ICLR_TO_NMI.md`](EXPERIMENT_PLAN_ICLR_TO_NMI.md) | Deprecated experiment plan | [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) |
| [`EVAL.md`](EVAL.md) | Deprecated `MECH_ET v3` evaluation guide | README plus the validation pipelines in [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) |
| [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) | Historical result inventory; not a citable result table | Frozen experiment manifests and the required-results section of the authoritative plan |
| [`README_DESIGN_NOTES.md`](README_DESIGN_NOTES.md) | Archived internal design notes | Current README |

## Documentation policy

1. The deterministic executor and `MECH_PROOF v1` semantics are described only in `PROOF_CARRYING.md`.
2. The experimental pipeline and the list of results required for a paper are described only in `PROOF_CENTRIC_EXPERIMENT_PLAN.md`.
3. Data leakage and benchmark-lineage rules may be expanded in `DATA_LEAKAGE_AND_ICLR_PLAN.md`, but model-training details must link back to the authoritative experiment plan.
4. Archived documents must begin with an explicit warning and a link to their replacement.
5. The README must not report unreleased checkpoints, pending benchmark numbers, or results from a non-matching split as validated performance.
6. Formal executability must never be described as energetic, kinetic, condition, or experimental validation.
