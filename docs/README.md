# MechET documentation map

This page distinguishes current method specifications from historical files and defines the reading order for collaborators.

## Reading order for collaborators

1. [`../README.md`](../README.md) — scientific story, implemented system and runnable entrypoints.
2. [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) — authoritative ICLR scientific and execution contract.
3. [`TRACE_FAITHFULNESS.md`](TRACE_FAITHFULNESS.md) — environment-owned trace, deterministic trace-to-proof compilation and causal endpoint contract.
4. [`TEXTBOOK_RAG.md`](TEXTBOOK_RAG.md) — natural-language textbook corpus, state-conditioned retrieval and bounded evidence cards.
5. [`MECHANISTIC_PRIMITIVE_LIBRARY.md`](MECHANISTIC_PRIMITIVE_LIBRARY.md) — Web evidence, executable anchor schema and structured model enhancement.
6. [`PROOF_CARRYING.md`](PROOF_CARRYING.md) — proof language and deterministic executor.
7. [`FORWARD_ELECTRON_EXPERT.md`](FORWARD_ELECTRON_EXPERT.md) — compact forward expert.
8. [`FRAMEWORK_MIGRATION.md`](FRAMEWORK_MIGRATION.md) — TRL, scale backends and Syntheseus planning.

## Authoritative documents

| Document | Purpose |
|---|---|
| [`../README.md`](../README.md) | Public architecture, current status, quickstart and method boundaries |
| [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) | Only source of truth for headline claims, required baselines, result tables, stopping gates and collaborator work packages |
| [`TRACE_FAITHFULNESS.md`](TRACE_FAITHFULNESS.md) | Main-method trace ownership, import semantics, `finish_trace`, compilation and faithfulness evaluation |
| [`TEXTBOOK_RAG.md`](TEXTBOOK_RAG.md) | Passage corpus, provenance schema, deterministic retrieval, evidence-card compilation and retrieval ablations |
| [`MECHANISTIC_PRIMITIVE_LIBRARY.md`](MECHANISTIC_PRIMITIVE_LIBRARY.md) | Source/license registry, executable anchor schema, structured retrieval, model augmentation and matched ablations |
| [`PROOF_CARRYING.md`](PROOF_CARRYING.md) | `MECH_PROOF v1`, deterministic executor and proof semantics |
| [`FORWARD_ELECTRON_EXPERT.md`](FORWARD_ELECTRON_EXPERT.md) | Forward data, model, training, generation, calibration and integration |
| [`FRAMEWORK_MIGRATION.md`](FRAMEWORK_MIGRATION.md) | Framework-neutral environment, TRL reference path, scale backends and planning migration |

## Current companion protocols

| Document | Purpose |
|---|---|
| [`PROOF_EQUIVALENCE.md`](PROOF_EQUIVALENCE.md) | Partial-order equivalence, proof-class deduplication, MechComp-OOD and failure certificates |
| [`DATA_LEAKAGE_AND_ICLR_PLAN.md`](DATA_LEAKAGE_AND_ICLR_PLAN.md) | Dataset lineage, benchmark freezing, overlap audit and decontamination |
| [`../knowledge/README.md`](../knowledge/README.md) | Mechanism source registry, download gates, raw evidence and knowledge-asset policy |
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
2. The main agent method must derive its proof and endpoint from the environment-owned trace; free-form proof submission is a baseline only.
3. Natural-language textbook passages, learned forward/selectivity scores and primitive-reference matches are soft evidence and must not be described as experimental truth.
4. Downloaded text remains a provenance-tracked passage; an LLM extraction becomes a released executable anchor only after explicit encoding and review.
5. Agent-training and planning frameworks must wrap the shared chemistry environments; they must not duplicate chemistry state or reward definitions.
6. Download revisions, hashes, licenses, corpus/index manifests, quarantine records, configs and calibration thresholds must accompany reported checkpoints.
7. Restricted sources require explicit license acceptance; no downloader may bypass upstream access terms.
8. The README must not report unreleased checkpoints or non-frozen numbers as validated performance.
9. Historical documents must retain clear deprecated/archived markers and links to replacements.
