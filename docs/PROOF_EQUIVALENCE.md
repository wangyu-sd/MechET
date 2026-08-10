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
+
a checkpoint trained only after the H2 split on H2/train
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
metadata.trace_plan.steps[].state_before
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

Build H2 from the frozen no-knowledge **training pool**, then train all H2 systems only on the resulting H2/train partition:

```bash
python scripts/build_mechcomp_ood.py \
  --input data/knowledge_ablation/v2/train/trace_no_knowledge.jsonl \
  --output-dir data/ood/mechcomp_source_sink \
  --test-fraction 0.10 \
  --valid-fraction 0.10 \
  --min-train-primitive-count 5 \
  --seed 42
```

Then train the trace-owned headline condition:

```bash
python scripts/train_tool_sft.py \
  --config configs/agent/tool_sft_mechcomp_trace.yaml
```

A checkpoint previously trained on the full pre-split `trace_no_knowledge` pool is invalid for H2, even if inference is later restricted to the H2 test set.

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
reaction_center_context_definition = step_state_plus_edge_imports_v2
```

## Headline claim gates

A headline H2 split requires all of the following:

| Gate | Requirement |
|---|---|
| Held-out data | Non-empty held-out test set |
| Composition separation | Zero train/test complete-composition overlap |
| Primitive coverage | Every test primitive appears in train at the declared minimum frequency |
| Training isolation | Headline checkpoint trained only on H2/train after split construction |
| Temporal integrity | Split frozen before final model evaluation |
| Representation fairness | Same stable IDs and data budget across baselines |
| Leakage audit | Product, reaction, scaffold, reaction-center, family, and near-duplicate overlap disclosed |

Holding out a family without controlling primitive coverage is a family-OOD experiment. Holding out unseen primitives is vocabulary extrapolation. Neither is the headline composition test.

## Structural overlap audit

Composition novelty can coexist with structural memorization. `build_mechcomp_ood.py` audits:

```text
exact structural product overlap
exact structural precursor overlap
exact full-reaction overlap
Murcko scaffold overlap
step-state reaction-center context overlap
reaction family overlap
Morgan/Tanimoto product near duplicates
```

Recommended strata:

```text
composition-OOD / scaffold-seen
composition-OOD / scaffold-unseen
composition-OOD / reaction-center-seen
composition-OOD / reaction-center-unseen
composition-OOD / family-seen
composition-OOD / family-unseen
```

A result confined to scaffold-seen examples supports recombination of known actions under familiar structures; it should not be presented as broad chemical extrapolation.

### Step-state reaction-center definition

The reaction center is defined where each inverse electron-flow move actually executes, not by searching all move atoms in the final product target.

For every trace step:

```text
current state = step.state_before + step.imports
  -> locate all source/sink atom maps for that step
  -> include their one-hop neighbors
  -> remove atom-map labels
  -> canonicalize local structural context
  -> combine with move topology and execution-primitive signatures
```

This matters because a chemically valid inverse step may import a precursor-side or auxiliary atom immediately before the move. Such an atom is legitimately absent from the product target. Product-only lookup previously misclassified these rows as `REACTION_CENTER_ATOMS_MISSING`; they must be recovered when the imported state resolves the atom maps. Only rows whose move atoms remain absent from the actual step state are quarantined.

The frozen definition identifier is:

```text
step_state_plus_edge_imports_v2
```

## Headline H2 inference

```bash
python scripts/run_h2_suite.py \
  --split-dir data/ood/mechcomp_source_sink \
  --adapter outputs/h2/tool_sft_trace_qwen3_0_6b \
  --out-dir outputs/h2 \
  --samples-per-target 4 \
  --seed 17
```

`run_h2_suite.py` verifies that the adapter manifest training SHA exactly matches `data/ood/mechcomp_source_sink/train.jsonl` before inference. The headline trace-owned condition uses `scripts/infer_mechet.py --mode trace`.

`scripts/infer_mechet_proof.py` is the independent complete-proof baseline only. Direct, free-form CoT/state-CoT, net-edit, complete-proof, and trace-owned systems must each be trained from the same frozen H2/train stable IDs before comparison.

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
step-state reaction-center novelty
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

For reaction-center auditing, report how many rows were previously rejected by the obsolete product-only lookup, how many are recovered by `step_state_plus_edge_imports_v2`, and which rows remain genuinely malformed. This audit is required before treating the earlier quarantine count as a chemistry-coverage statistic.

## Interpretation boundary

H2 supports a compositional-generalization claim only when known source-to-sink units form unseen complete programs under audited structural overlap and a checkpoint that never saw the held-out compositions. It does not establish a unique physical mechanism or experimental feasibility.
