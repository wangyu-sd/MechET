# Documentation map

MechET has one scientific story:

> **A retrosynthetic mechanism is treated as a causal, executable program over source-to-sink electron-flow primitives, then tested for causal use, compositional generalization, and evidence benefit.**

This page defines the authority order and the shortest reading path for different audiences.

## Start here

| Reader | Recommended path | Purpose |
|---|---|---|
| **Editor or reviewer** | `README` → `STATUS_MATRIX` → `SCIENTIFIC_THESIS` → `PROOF_CENTRIC_EXPERIMENT_PLAN` | Separate implemented infrastructure from established scientific evidence |
| **Experimental lead** | `EXECUTION_PLAN` → `TOOL_SFT` → H1/H2/H3 topic documents | Run the study in the correct order and stop at failed gates |
| **Runtime developer** | `TRACE_FAITHFULNESS` → `PROOF_CARRYING` → `FRAMEWORK_MIGRATION` | Preserve the causal endpoint path and tool surface |
| **Data/evidence curator** | `KNOWLEDGE_ABLATIONS` → `TEXTBOOK_RAG` → `MECHANISTIC_PRIMITIVE_LIBRARY` | Build frozen evidence conditions with provenance, quality metadata, and controls |
| **Planning researcher** | Core documents first → `FORWARD_ELECTRON_EXPERT` → planning adapters | Treat planning as a downstream extension, not evidence for H1/H2 |

## Authority order

1. [`SCIENTIFIC_THESIS.md`](SCIENTIFIC_THESIS.md) — scientific question, hypotheses, terminology, claim ladder, and prohibited interpretations.
2. [`TRACE_FAITHFULNESS.md`](TRACE_FAITHFULNESS.md) — main causal runtime contract and H1 intervention semantics.
3. [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md) — paper-level claim–evidence matrix and integrity gates.
4. [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md) — ordered commands, required artifacts, stopping conditions, and handoff criteria.
5. [`TOOL_SFT.md`](TOOL_SFT.md) — replay-verified supervision, tokenizer audit, learnability gate, and adapter lineage.
6. [`PROOF_EQUIVALENCE.md`](PROOF_EQUIVALENCE.md) — H2 source-to-sink execution primitives, composition signatures, and structural-overlap audit.
7. [`KNOWLEDGE_ABLATIONS.md`](KNOWLEDGE_ABLATIONS.md) — H3 matched evidence conditions, interventions, and claim gates.
8. [`TEXTBOOK_RAG.md`](TEXTBOOK_RAG.md) — corpus provenance, bounded retrieval, source-quality metadata, and evidence-card contract.
9. [`MECHANISTIC_PRIMITIVE_LIBRARY.md`](MECHANISTIC_PRIMITIVE_LIBRARY.md) — structured mechanistic knowledge anchors.
10. [`PROOF_CARRYING.md`](PROOF_CARRYING.md) — `MECH_PROOF v1`, execution semantics, and complete-proof baselines.
11. [`FORWARD_ELECTRON_EXPERT.md`](FORWARD_ELECTRON_EXPERT.md) — optional independent forward evidence and calibration requirements.
12. [`FRAMEWORK_MIGRATION.md`](FRAMEWORK_MIGRATION.md) — framework adapters and downstream planning integration.

[`STATUS_MATRIX.md`](STATUS_MATRIX.md) is a living implementation/evidence status page, not a competing scientific source of truth.

When documents disagree, update the lower-authority document. Do not create a parallel source of truth.

## Scientific architecture

```text
scientific bottleneck
  reasoning may not cause the answer
        |
        v
trace-owned formulation
  explicit tool actions -> environment-owned trace -> finish_trace
        |
        v
formal verification
  replay declared moves -> compile proof -> execute endpoint
        |
        +-----------------------------+
        |                             |
        v                             v
H1 causal-use test              H2 composition test
observation interventions       known primitives / unseen compositions
        |
        v
H3 evidence test
frozen textbook/anchors vs matched controls
```

## Runtime contract

The main model-facing implementation is an explicit TRL facade: `TraceOwnedTRLEnvironment` or a declared evidence variant.

```text
model tool calls
  -> environment-owned state transitions
  -> source-to-sink execution primitives
  -> committed trace
  -> replay declared moves
  -> finish_trace
  -> deterministic proof compilation
  -> executor-derived endpoint views
```

The main facade exposes declared tools only. Internal methods such as `state_dict` remain private, and `submit_proof` is available only in a named legacy baseline.

Root imports and edge imports survive proof-to-trace conversion, replay, and compilation. Every valid, invalid, unavailable, or disabled call consumes the same frozen tool budget.

## Data and artifact contract

### Supervision rows

Tool-SFT rows contain:

```text
messages and tools
JSON-object arguments
one result per tool call
exactly one successful finish_trace
frozen endpoint views
trace and move-sequence digests
executor replay metadata
```

Reaction-family labels are excluded from headline retrieval queries. `label_oracle` is an upper bound only.

### Prediction artifacts

Prediction artifacts are distinct from supervision rows. Headline evaluation requires:

```text
artifact_type=prediction
frozen reference ID universe
complete model/tokenizer/adapter revision metadata
global and per-candidate seeds
generation and selector contract
raw rollout state and terminal result
```

Missing predictions remain in the denominator as failures. Duplicate or extra IDs are errors. Trace metrics require an explicit successful `finish_trace` and are recomputed rather than trusted from stored booleans.

## Statistical contract

H1 and H3 primary contrasts now report:

```text
paired bootstrap confidence intervals over frozen target IDs
exact McNemar tests over discordant target pairs
Holm family-wise error correction over declared contrast families
independent-seed aggregation with seed-level bootstrap intervals
```

A point estimate above zero is not sufficient for a final claim. `scripts/aggregate_evaluation_seeds.py` combines independent H1 or H3 evaluation artifacts after each seed has passed its own runtime and pairing checks.

## Source and documentation integrity

External sources are governed by `knowledge/source_registry.yaml`, which records license, redistribution policy, quality status, permitted uses, disallowed uses, and optional page-level overrides.

- `scripts/check_source_health.py` checks configured URLs, resolved MediaWiki titles, revisions, non-empty content, soft 404s, redirects, hashes, and quality warnings.
- `scripts/check_documentation_integrity.py` checks internal links, Markdown anchors, image paths, documented scripts, documented configs, script syntax, and CLI entrypoint structure.
- The scheduled source-health workflow is non-blocking for normal PRs and opens or updates a repository issue when external sources fail.

## Scientific terminology

### Source-to-sink execution primitives

Local executable actions such as `LP -> BOND`, `BOND -> ATOM`, and `BOND -> BOND`. They define the H2 composition basis.

### Mechanistic knowledge anchors

Provenance-aware records with role bindings, candidate moves, warnings, competitors, and references. They are soft evidence and do not define the H2 primitive vocabulary.

### Formal and empirical evidence

The deterministic executor defines formal validity. Textbook passages, anchor matches, and learned forward scores are soft evidence and cannot override formal failure or establish experimental truth.

## Claim ladder

```text
L0 infrastructure executes
  -> L1 tool contract is learnable
  -> L2 tool observations are causally used
  -> L3 known primitives compose out of distribution
  -> L4 external evidence adds information
  -> L5 verified programs improve downstream search/planning
```

A later level cannot rescue an earlier failure.

## Experimental order

1. Measure proof-to-trace conversion, budget compatibility, and quarantine coverage.
2. Build the six matched Tool-SFT conditions.
3. Run real tokenizer/mask audits and a small-set overfit.
4. Evaluate H1 under identical runtime contracts and audited observation interventions.
5. Evaluate H2 on a non-empty, composition-disjoint split whose primitives all occur in train, then report product, precursor, reaction, scaffold, center, family, and near-duplicate overlap.
6. Evaluate H3 with frozen evidence, irrelevant context, direct open-book, and evidence-content interventions.
7. Aggregate independent seeds and require confidence-interval and corrected-test gates.
8. Scale models or add RL only after pilot gates pass.
9. Treat planning as a downstream extension.

## Companion documents

- [`STATUS_MATRIX.md`](STATUS_MATRIX.md) — implementation status versus unestablished scientific results.
- [`KNOWLEDGE_AUGMENTED_AGENT.md`](KNOWLEDGE_AUGMENTED_AGENT.md) — evidence-tool implementation details.
- [`DATA_LEAKAGE_AND_ICLR_PLAN.md`](DATA_LEAKAGE_AND_ICLR_PLAN.md) — overlap audits and benchmark freezing.
- [`../knowledge/README.md`](../knowledge/README.md) — source licensing, quality metadata, and asset policy.
- [`../data/README.md`](../data/README.md) — local data construction.

## Historical documents

Archived or deprecated documents remain visibly marked:

- [`EXPERIMENT_PLAN_ICLR_TO_NMI.md`](EXPERIMENT_PLAN_ICLR_TO_NMI.md) — deprecated.
- [`EVAL.md`](EVAL.md) — deprecated legacy evaluation guide.
- [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) — historical inventory, not a result table.
- [`README_DESIGN_NOTES.md`](README_DESIGN_NOTES.md) — archived internal notes.

## CI documentation contract

`tests/test_documentation_contract.py` protects the semantic contract rather than exact prose. `scripts/check_documentation_integrity.py` protects executable references. Together they check that documentation:

- retains the explicit TRL facade and private internal methods;
- preserves root imports and move replay;
- distinguishes source-to-sink execution primitives from knowledge anchors;
- separates supervision from prediction artifacts;
- requires explicit `finish_trace` for trace credit;
- uses Pass@K rather than mislabeling unranked generations as Top-K;
- references existing scripts, configs, files, images, and anchors;
- presents planning as a downstream extension.
