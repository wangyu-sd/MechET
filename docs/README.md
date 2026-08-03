# MechET documentation map

This page defines the authority order for MechET. The repository must have one scientific story: causal program induction over electron-flow execution primitives, followed by tests of compositional generalization and evidence separation.

## Reading order

1. [`SCIENTIFIC_THESIS.md`](SCIENTIFIC_THESIS.md) — single source of truth for the scientific question, hypotheses, terminology and permitted claims.
2. [`TRACE_FAITHFULNESS.md`](TRACE_FAITHFULNESS.md) — main-method causal contract: environment-owned trace, `finish_trace`, proof compilation and endpoint derivation.
3. [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) — authoritative definitions of H1 causal faithfulness, H2 compositionality and H3 evidence separation.
4. [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md) — ordered data, training, intervention and evaluation commands with stop conditions.
5. [`TOOL_SFT.md`](TOOL_SFT.md) — conservative proof-to-trace conversion, replay-verified supervision and coverage reporting.
6. [`KNOWLEDGE_ABLATIONS.md`](KNOWLEDGE_ABLATIONS.md) — matched evidence conditions, fair direct open-book baseline and causal interventions.
7. [`MECHANISTIC_PRIMITIVE_LIBRARY.md`](MECHANISTIC_PRIMITIVE_LIBRARY.md) — provenance-aware mechanistic knowledge anchors; not the execution-primitive basis used for composition-OOD.
8. [`PROOF_CARRYING.md`](PROOF_CARRYING.md) — `MECH_PROOF v1`, deterministic executor and complete-proof baseline.
9. [`FORWARD_ELECTRON_EXPERT.md`](FORWARD_ELECTRON_EXPERT.md) — optional independent process, competitor and uncertainty evidence.
10. [`FRAMEWORK_MIGRATION.md`](FRAMEWORK_MIGRATION.md) — implementation backends and planning adapters; not a scientific definition.

## Authority

| Document | Authority |
|---|---|
| `SCIENTIFIC_THESIS.md` | scientific question, hypothesis, terminology and claim boundaries |
| `TRACE_FAITHFULNESS.md` | main inference contract and the meaning of causal trace ownership |
| `PROOF_CENTRIC_EXPERIMENT_PLAN.md` | frozen experiment definitions and claim gates |
| `EXECUTION_PLAN.md` | run order, commands, pilot gates and stop conditions |
| `README.md` | public summary and entrypoints; must follow the four documents above |

When documents disagree, update the lower-authority document. Do not create a new parallel source of truth.

## Main method and baselines

### Main method

```text
TraceOwnedAgentEnv / KnowledgeAugmentedAgentEnv
explicit electron-flow actions
finish_trace
deterministic trace-to-proof compilation
executor-derived precursor
```

`submit_proof` is disabled.

### Required baselines

```text
outcome-only
free-form CoT plus answer
state-CoT plus answer
net edit
independent complete MECH_PROOF generation
legacy loose tool trace plus submitted proof
```

Complete-proof generation remains supported but must not be described as the trace-owned main method.

## Terminology policy

### Electron-flow execution primitive

A local executable source-to-sink action. Execution primitives define the action vocabulary and MechComp-OOD composition signatures.

### Mechanistic knowledge anchor

A curated record with structural patterns, role bindings, candidate moves, warnings, competitors and provenance. Knowledge anchors are soft evidence and must not be called the compositional primitive basis without qualification.

### Soft evidence

Textbook passages, structured knowledge-anchor matches and learned forward scores. Soft evidence cannot override the deterministic executor or establish experimental truth.

## Current companion documents

| Document | Purpose |
|---|---|
| [`TEXTBOOK_RAG.md`](TEXTBOOK_RAG.md) | passage corpus, provenance, retrieval and bounded evidence cards |
| [`KNOWLEDGE_AUGMENTED_AGENT.md`](KNOWLEDGE_AUGMENTED_AGENT.md) | online evidence tools and trace integration |
| [`PROOF_EQUIVALENCE.md`](PROOF_EQUIVALENCE.md) | partial-order equivalence, invariance and composition signatures |
| [`DATA_LEAKAGE_AND_ICLR_PLAN.md`](DATA_LEAKAGE_AND_ICLR_PLAN.md) | benchmark freezing, overlap audit and decontamination |
| [`../knowledge/README.md`](../knowledge/README.md) | source registry, download gates and knowledge-asset policy |
| [`../data/README.md`](../data/README.md) | proof-data construction details |
| [`../data/FORWARD_EXPERT.md`](../data/FORWARD_EXPERT.md) | forward-expert local data layout |

## Archived or deprecated documents

| Document | Status | Replacement |
|---|---|---|
| [`EXPERIMENT_PLAN_ICLR_TO_NMI.md`](EXPERIMENT_PLAN_ICLR_TO_NMI.md) | deprecated | `PROOF_CENTRIC_EXPERIMENT_PLAN.md` and `EXECUTION_PLAN.md` |
| [`EVAL.md`](EVAL.md) | deprecated legacy `MECH_ET v3` guide | current method and experiment documents |
| [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) | historical inventory; not a result table | frozen manifests and result artifacts |
| [`README_DESIGN_NOTES.md`](README_DESIGN_NOTES.md) | archived internal notes | current README and scientific thesis |

## Documentation policy

1. The main method ends with `finish_trace`; free-form proof submission is a baseline only.
2. The environment-owned trace is the sole source of the main-method proof and endpoint.
3. Electron-flow execution primitives and mechanistic knowledge anchors are distinct concepts.
4. The deterministic executor is the hard source of formal validity.
5. Textbook retrieval, knowledge-anchor matches and forward scores are soft evidence.
6. Knowledge retrieval must not receive direct reward or return an endpoint.
7. Tool-SFT rows must replay through the same trace-owned environment used at inference.
8. Ambiguous source/sink labels are quarantined rather than invented.
9. Headline comparisons use matched IDs, targets, structural endpoints and declared budgets.
10. Anchors-only and direct open-book conditions are derived from the same matched source rows, not manually assembled datasets.
11. Tool-observation interventions are required for the causal reasoning claim.
12. Composition-OOD is defined over execution primitives, not knowledge-anchor IDs.
13. Paper-scale RL begins only after Tool-SFT demonstrates executable learning.
14. Forward evidence is independently calibrated before use in ranking or reward.
15. Planning is a downstream extension and cannot rescue failed causal or compositional claims.
16. Source revisions, hashes, licenses, configs, seeds, checkpoint lineage and raw predictions accompany reported results.
17. The README must not report unreleased checkpoints or non-frozen numbers as established results.

## CI documentation contract

`tests/test_documentation_contract.py` prevents the following regressions:

- restoring the old bidirectional-system story as the primary thesis;
- describing `submit_proof` as the main terminal action;
- omitting `finish_trace` or trace ownership;
- conflating execution primitives with knowledge anchors;
- presenting evidence or planning extensions as the causal endpoint path;
- pointing default commands to the legacy loose-trace baseline.
