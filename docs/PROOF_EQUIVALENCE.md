# H2 source-to-sink composition generalization

## Scientific hypothesis

H2 tests whether familiar local electron-flow execution primitives can be recombined into complete move compositions absent from training.

The primitive basis is the explicit model-facing move vocabulary:

```text
LP -> BOND
BOND -> ATOM
BOND -> BOND
```

with mapped source/sink role features. Mechanistic knowledge-anchor IDs and net MECH_PROOF bond/charge deltas do not define the headline split.

## Required input

`build_mechcomp_ood.py` accepts replay-verified Tool-SFT rows containing:

```text
metadata.executor_replayed = true
metadata.trace_plan.initial_imports
metadata.trace_plan.steps[].moves
```

Rows without a valid explicit trace plan are quarantined.

## Primitive and composition signatures

Each source-to-sink move is canonicalized using:

```text
source kind
sink kind
source and sink atom-element/charge/aromatic features
bond context when present
electron count
```

Original atom-map labels are excluded. A complete composition signature includes the ordered elementary steps while treating coupled moves within one step as a deterministic multiset.

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

The manifest reports:

```text
primitive_basis = source_to_sink_execution_moves_v1
eligible and quarantined rows
train/valid/test sizes
requested and achieved fractions
composition overlap
held-out primitive coverage
minimum train primitive count
```

## Claim gates

A headline H2 split requires:

```text
non-empty held-out test set
zero train/test complete-composition overlap
every test primitive appears in train at the declared minimum frequency
split generated before final model evaluation
same stable IDs across representation baselines
```

Holding out a reaction family without controlling primitive coverage is a different OOD test. Holding out unseen primitives does not test composition of known units.

## Representation comparisons

Evaluate matched:

```text
outcome-only direct generation
free-form CoT
state-CoT
reaction center / synthon when labels exist
net edit
independent complete proof
trace-owned source-to-sink Tool-CoT
```

Report performance versus:

```text
composition frequency
number of elementary steps
number of source-to-sink moves
proof topology
changed atoms and bonds
ring formation/change
reaction family
product scaffold
```

## Proof equivalence remains a separate metric

`MECH_PROOF v1` partial-order equivalence still canonicalizes executable bond/lone-pair/charge programs modulo state IDs, map labels, and commuting independent events. It is useful for proof-class deduplication and complete-proof baselines, but it is not the H2 execution-primitive split definition.

## Failure and repair

Structured executor failures remain available for analysis and bounded repair. Any repaired result must be re-executed and retain its original frozen example ID; repaired test examples cannot be used to alter the split or primitive vocabulary.
