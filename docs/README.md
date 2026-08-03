# MechET documentation map

This page defines the authority order for MechET. The repository has one scientific story: causal program induction over source-to-sink execution primitives, followed by tests of compositional generalization and evidence separation.

## Authority and reading order

1. [`SCIENTIFIC_THESIS.md`](SCIENTIFIC_THESIS.md) — scientific question, hypotheses, terminology and permitted claims.
2. [`TRACE_FAITHFULNESS.md`](TRACE_FAITHFULNESS.md) — main causal runtime contract.
3. [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) — frozen H1/H2/H3 experiment definitions and claim gates.
4. [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md) — ordered commands, artifacts and stopping rules.
5. [`TOOL_SFT.md`](TOOL_SFT.md) — replay-verified supervision and adapter lineage.
6. [`PROOF_EQUIVALENCE.md`](PROOF_EQUIVALENCE.md) — source-to-sink primitive and composition signatures.
7. [`KNOWLEDGE_ABLATIONS.md`](KNOWLEDGE_ABLATIONS.md) — matched evidence conditions and interventions.
8. [`TEXTBOOK_RAG.md`](TEXTBOOK_RAG.md) — corpus, provenance, retrieval and bounded evidence cards.
9. [`MECHANISTIC_PRIMITIVE_LIBRARY.md`](MECHANISTIC_PRIMITIVE_LIBRARY.md) — structured mechanistic knowledge anchors.
10. [`PROOF_CARRYING.md`](PROOF_CARRYING.md) — `MECH_PROOF v1` and complete-proof baselines.
11. [`FORWARD_ELECTRON_EXPERT.md`](FORWARD_ELECTRON_EXPERT.md) — optional independent forward evidence.
12. [`FRAMEWORK_MIGRATION.md`](FRAMEWORK_MIGRATION.md) — implementation backends and downstream planning adapters.

When documents disagree, update the lower-authority document. Do not create a parallel source of truth.

## Runtime contract

The main model-facing implementation is an explicit TRL facade (`TraceOwnedTRLEnvironment` or an evidence variant). It exposes only declared tools. Internal state inspection such as `state_dict` remains private, and `submit_proof` is available only in a named legacy baseline.

```text
model tool calls
  -> environment-owned state transitions
  -> source-to-sink move trace
  -> replay declared moves
  -> finish_trace
  -> deterministic proof compilation
  -> executor-derived endpoint views
```

Root imports and edge imports must survive proof-to-trace conversion and replay. Every invalid or disabled tool call consumes the same environment budget used by valid calls.

## Data and artifact contract

Tool-SFT rows use paired `messages` and `tools`, JSON-object arguments, a frozen tokenizer assistant mask and explicit endpoint views. Reaction-family labels are excluded from headline retrieval queries; a `label_oracle` query is only an upper bound.

Prediction artifacts are distinct from supervision rows. Headline evaluation requires `artifact_type=prediction`, a frozen reference ID universe and recorded model, adapter, revision and generation settings. Missing predictions stay in the denominator; duplicate or extra IDs are errors. Trace and proof metrics are recomputed instead of trusting stored booleans.

## Scientific terminology

### Source-to-sink execution primitives

Local executable actions such as `LP -> BOND`, `BOND -> ATOM` and `BOND -> BOND`. These define the H2 composition split.

### Mechanistic knowledge anchors

Provenance-aware records with role bindings, candidate moves, warnings and competitors. They are soft evidence and do not define the H2 primitive basis.

### Formal and empirical evidence

The deterministic executor defines formal validity. Textbook passages, anchor matches and learned forward scores are soft evidence and cannot override formal failure or establish experimental truth.

## Main method and baselines

Main method:

```text
trace-owned tool reasoning
finish_trace
environment-compiled proof
executor-derived endpoint
```

Required baselines include direct outcome generation, answer-bearing CoT, independent complete-proof generation and legacy loose-trace plus submitted proof.

## Experimental order

1. Measure proof-to-trace conversion and quarantine coverage.
2. Build the six matched Tool-SFT conditions.
3. Run real tokenizer/mask audits and small-set overfit tests.
4. Evaluate H1 with normal, removed, stale and shuffled tool observations under the same runtime contract.
5. Evaluate H2 with non-empty, composition-disjoint test splits whose primitives all occur in train.
6. Evaluate H3 with frozen textbook/anchor evidence, irrelevant text, direct open-book and evidence-content interventions.
7. Scale models or add RL only after the pilot gates pass.

Planning is a downstream extension and cannot rescue failed causal, compositional or evidence claims.

## Companion and historical documents

- [`KNOWLEDGE_AUGMENTED_AGENT.md`](KNOWLEDGE_AUGMENTED_AGENT.md) describes evidence tools.
- [`DATA_LEAKAGE_AND_ICLR_PLAN.md`](DATA_LEAKAGE_AND_ICLR_PLAN.md) describes overlap audits and benchmark freezing.
- [`../knowledge/README.md`](../knowledge/README.md) describes source licensing and asset policy.
- [`../data/README.md`](../data/README.md) describes local data construction.

Archived or deprecated documents remain visibly marked:

- [`EXPERIMENT_PLAN_ICLR_TO_NMI.md`](EXPERIMENT_PLAN_ICLR_TO_NMI.md) — deprecated.
- [`EVAL.md`](EVAL.md) — deprecated legacy evaluation guide.
- [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) — historical inventory, not a result table.
- [`README_DESIGN_NOTES.md`](README_DESIGN_NOTES.md) — archived internal notes.

## CI documentation contract

`tests/test_documentation_contract.py` prevents restoration of the old system story, exposure of internal methods as model tools, omission of root imports or prediction artifacts, conflation of execution primitives with knowledge anchors, and presentation of planning as the causal endpoint path.