# Partial-order equivalence and execution-primitive composition OOD

This document defines the semantic comparison and split construction used to test H2:

> Can familiar electron-flow execution primitives be composed into mechanisms not observed during training?

Mechanistic knowledge-anchor IDs are not used to define the headline composition split.

## Why exact string matching is insufficient

Two executable programs may be semantically equivalent while differing in:

```text
state identifiers
textual edge order
atom-map labels
component order
ordering of independent events that touch disjoint atoms
```

Evaluation therefore uses canonical partial-order signatures rather than proof-text equality.

## Canonical event signature

For each executable event, MechET derives a map-label-invariant signature from:

```text
elements and initial formal charges
imported-atom roles
bond-order changes
lone-pair changes
charge transitions
non-commuting dependencies
```

Events touching disjoint atom sets commute. Dependencies are retained when events share mapped atoms or state requirements.

```python
from mechet import canonical_partial_order_signature, proofs_equivalent

signature = canonical_partial_order_signature(proof_text)
equivalent = proofs_equivalent(predicted_proof, reference_proof)
```

Report separately:

```text
proof_equivalent_to_gold
execution_primitive_composition_match
structural_endpoint_match
```

A different executable proof may reach the same endpoint without being mechanism-equivalent; a different endpoint is not automatically chemically wrong.

## Execution primitive versus knowledge anchor

### Execution primitive

A local formal action/delta pattern used to construct and compare executable programs. It defines H2 composition coverage.

### Mechanistic knowledge anchor

A curated retrieval record with structural roles, candidate actions, warnings and provenance. Anchor IDs may leak higher-level reaction-family information and therefore cannot define MechComp-OOD.

## MechComp-OOD construction

`build_mechcomp_ood.py` holds out complete execution-primitive compositions while requiring every constituent primitive in validation/test to remain represented in training.

```bash
python scripts/build_mechcomp_ood.py \
  --input data/mechet_proof_clean/train.jsonl \
  --output-dir data/mechet_proof_mechcomp \
  --test-fraction 0.10 \
  --valid-fraction 0.10 \
  --min-train-primitive-count 5 \
  --seed 42
```

The frozen manifest must report:

```text
train/valid/test IDs and hashes
zero complete-composition overlap
one hundred percent held-out constituent-primitive coverage
minimum train count per held-out primitive
composition frequency distribution
proof length and topology distribution
family and scaffold distribution
```

## Split validity checks

A headline H2 split is invalid when:

- a test execution primitive is absent from training;
- composition signatures use knowledge-anchor IDs or reaction names;
- train/test overlap remains at exact reaction, product, center or patent-family levels without disclosure;
- test compositions are selected after model evaluation;
- complexity distributions are so different that composition novelty cannot be separated from size alone.

## Required matched comparisons

```text
outcome-only direct generation
free-form CoT plus answer
state-CoT plus answer
net-edit generation
independent complete proof
trace-owned Tool-CoT
trace-owned Tool-CoT plus external evidence
```

Use identical stable IDs, structural endpoints, base-model families, updates and seeds where applicable.

## Required analyses

Report H2 performance by:

```text
composition frequency in the source corpus
minimum constituent-primitive frequency
proof length
number of changed atoms and bonds
imports
ring formation or ring change
stereochemical change
chain/tree/DAG topology
reaction family
product scaffold
```

The key result is not merely an aggregate OOD number. The analysis should reveal when the execution vocabulary supports recombination and where coverage or topology causes failure.

## Representation invariance

Test synchronized transformations:

```text
atom-map permutation
state-ID renaming
edge serialization
component ordering
reordering of commuting independent events
verified equivalent proof variants
```

Semantic robustness is measured by execution, endpoint and partial-order equivalence, not exact text equality.

## Failure certificates and repair

`diagnose_proof` maps the first deterministic failure to a stable code and action/edge location, including parse, atom-map, bond, lone-pair, charge, conservation, sanitization, reachability and DAG-join failures.

```python
from mechet import diagnose_proof, format_repair_feedback

certificate = diagnose_proof(prediction)
feedback = format_repair_feedback(certificate) if certificate else "OK"
```

Only semantics-preserving declaration corrections may be deterministic. Bond, charge, import and dependency changes require a new model proposal followed by complete execution.

## Extended evaluation

```bash
python scripts/eval_mechet_proof_generations.py \
  --data data/mechet_proof_mechcomp/test.jsonl \
  --predictions outputs/mechet_proof_eval/generations.jsonl \
  --attempt-local-repair \
  --out outputs/mechet_proof_eval/summary.json
```

Report:

```text
FormatPass and ExecutePass
structural endpoint accuracy
partial-order equivalence
execution-primitive composition match
failure-code distribution
repair success and new-error introduction
metrics by composition novelty and topology
```

## Claim boundary

H2 supports a compositional-reasoning claim only when test compositions are unseen, all constituent execution primitives are seen, and trace/proof models outperform matched alternatives in a way not explained solely by reaction family, scaffold or complexity shift.
