# H2 — source-to-sink composition generalization

> **Question:** can familiar local execution primitives be recombined into complete mechanisms absent from training?  
> **Primitive basis:** `source_to_sink_execution_moves_v1`  
> **Non-goal:** holding out unseen primitives or merely holding out reaction-family labels

## Identification target

H2 isolates **composition novelty** from vocabulary novelty.

A positive result requires:

```text
known constituent execution primitives
+
unseen complete move composition
+
frozen structural-overlap audits
```

It does not ask whether a model can extrapolate to an unseen electron-flow primitive, nor whether it can recognize a reaction-name label absent from training.

## Execution primitive basis

The primitive vocabulary is defined by explicit model-facing source-to-sink actions:

```text
LP -> BOND
BOND -> ATOM
BOND -> BOND
```

with mapped source/sink role features and local chemical context.

Mechanistic knowledge-anchor IDs and net `MECH_PROOF v1` bond, lone-pair, or charge deltas do not define the headline H2 split.

## Required input contract

`build_mechcomp_ood.py` accepts replay-verified Tool-SFT rows containing:

```text
metadata.executor_replayed = true
metadata.trace_plan.initial_imports
metadata.trace_plan.steps[].imports
metadata.trace_plan.steps[].moves
metadata.endpoint_source = environment_owned_trace
```

Rows lacking a valid explicit trace plan are quarantined rather than assigned to a split.

## Primitive signature

Each source-to-sink move is canonicalized from:

```text
source kind
sink kind
source atom element, formal charge, and aromaticity
sink atom element, formal charge, and aromaticity
bond context where present
electron count
```

Original atom-map labels are excluded so signatures represent chemical roles rather than example-specific identifiers.

## Composition signature

A complete composition signature records:

- the ordered sequence of elementary transition steps;
- imports associated with each transition;
- the deterministic multiset of coupled moves within a step;
- the canonical primitive signatures for all moves.

Coupled actions within one atomic step are order-invariant. Sequential steps remain ordered because they operate on different molecular states.

## Build the split

```bash
python scripts/build_mechcomp_ood.py \
  --input data/knowledge_ablation/v2/trace_no_knowledge.jsonl \
  --output-dir data/ood/mechcomp_source_sink \
  --test-fraction 0.10 \
  --valid-fraction 0.10 \
  --min-train-primitive-count 5 \
  --seed 42
```

## Split manifest

The manifest must report:

```text
primitive_basis = source_to_sink_execution_moves_v1
eligible and quarantined rows
train, validation, and test sizes
requested and achieved fractions
complete-composition overlap
held-out primitive coverage
minimum train primitive count
seed and manifest hash
```

## Headline claim gates

A headline H2 split requires all of the following:

| Gate | Requirement |
|---|---|
| Held-out data | Non-empty held-out test set |
| Composition separation | Zero train/test complete-composition overlap |
| Primitive coverage | Every test primitive appears in train at the declared minimum frequency |
| Temporal integrity | Split frozen before final model evaluation |
| Representation fairness | Same stable IDs and data budget across baselines |
| Leakage audit | Product, reaction, scaffold, and reaction-center overlap disclosed |

Holding out a family without controlling primitive coverage is a family-OOD experiment. Holding out unseen primitives is vocabulary extrapolation. Neither is the headline composition test.

## Structural overlap audit

Composition novelty can coexist with structural memorization. Report at least:

```text
exact structural product overlap
exact structural reaction overlap
product scaffold similarity
reaction-center template overlap
precursor-pair overlap
reaction family
ring formation or ring change
```

Recommended strata:

```text
composition-OOD / scaffold-seen
composition-OOD / scaffold-unseen
composition-OOD / family-seen
composition-OOD / family-unseen
```

A result confined to scaffold-seen examples supports recombination of known actions under familiar structures; it should not be presented as broad chemical extrapolation.

## Representation comparisons

Evaluate matched:

```text
outcome-only direct generation
free-form CoT
state-CoT
reaction-center or synthon prediction when frozen labels exist
net edit
independent complete proof
trace-owned source-to-sink Tool-CoT
```

All systems use the same examples, model family, frozen revision, optimization budget, and endpoint definitions.

## Required reporting axes

Report performance against:

```text
composition frequency in training
composition novelty rank
number of elementary steps
number of source-to-sink moves
proof topology
changed atoms and bonds
ring formation/change
reaction family
product scaffold similarity
```

Primary metrics are StructuralEndpointPass@K and formal execution metrics. Mapped exact remains secondary.

## Proof equivalence is a separate object

`MECH_PROOF v1` partial-order equivalence canonicalizes executable bond/lone-pair/charge programs modulo state IDs, atom-map labels, and commuting independent events.

It is useful for:

- proof-class deduplication;
- complete-proof baselines;
- equivalent-program analysis;
- bounded augmentation through valid topological orderings.

It is **not** the H2 execution-primitive split definition.

## Equivalent-trace augmentation

When independent transitions commute, limited valid topological orderings may be used for training augmentation only if:

1. every ordering replays through the executor;
2. all variants compile to the same proof-equivalence class;
3. train/test split assignment occurs before augmentation;
4. no test-equivalent variant enters training.

## Failure and repair

Structured executor failures remain available for analysis and bounded repair. Any repaired result must:

- preserve the original frozen example ID;
- re-execute successfully;
- retain the original split assignment;
- never alter the primitive vocabulary or held-out composition definition.

## Interpretation boundary

H2 supports a compositional-generalization claim only when known source-to-sink units form unseen complete programs under audited structural overlap. It does not establish a unique physical mechanism or experimental feasibility.

## Automated structural-overlap audit

`build_mechcomp_ood.py` now audits exact product, structural-precursor and full reaction overlap; Murcko scaffold overlap; reaction-center context overlap; reaction-family overlap; and Morgan/Tanimoto near duplicates. Valid and test rows receive `metadata.mechcomp_structural_overlap`, and the manifest reports composition-OOD strata for scaffold-seen/unseen, center-seen/unseen and family-seen/unseen subsets. Composition novelty must not be interpreted as scaffold or family novelty.
