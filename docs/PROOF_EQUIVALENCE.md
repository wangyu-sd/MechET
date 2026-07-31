# Partial-order proof equivalence and MechComp-OOD

This layer extends `MECH_PROOF v1` from executable proof generation to evaluation and data splitting over mechanism equivalence classes.

## Why exact trace matching is insufficient

Two executable proofs may describe the same mechanism while differing in:

- internal state identifiers;
- textual `EDGE` order;
- arbitrary atom-map labels;
- the order of electron-flow events that touch disjoint atoms.

The evaluator therefore constructs a canonical partial-order signature rather than comparing proof text.

## Canonical signature

For every proof edge, MechET derives a map-label-invariant primitive signature from:

- element and initial formal charge;
- whether an atom is introduced by `IMPORT`;
- bond-order changes;
- lone-pair changes;
- charge transitions.

Proof-path precedence is retained only when two events touch a common atom map. Disjoint events are treated as commuting. The full signature also contains the canonical target, root imports and executor-derived endpoint.

```python
from mechet import canonical_partial_order_signature, proofs_equivalent

signature = canonical_partial_order_signature(proof_text)
equivalent = proofs_equivalent(predicted_proof, reference_proof)
```

This provides two complementary evaluation levels:

- `proof_equivalent_to_gold`: same target, imports, endpoint, event multiset and non-commuting dependencies;
- `composition_match`: same mechanism primitive composition, excluding target and endpoint molecules.

## MechComp-OOD split

`build_mechcomp_ood.py` holds out complete mechanism compositions while requiring every primitive in valid/test to remain represented in train.

```bash
python scripts/build_mechcomp_ood.py \
  --input data/mechet_proof_sft/train.jsonl \
  --output-dir data/mechet_proof_mechcomp \
  --test-fraction 0.10 \
  --valid-fraction 0.10 \
  --min-train-primitive-count 5 \
  --seed 42
```

The manifest reports:

- train/valid/test sizes;
- composition overlap, which should be zero;
- held-out primitive coverage, which should be one;
- the minimum remaining train count for any primitive.

The scientific question is whether a model can assemble familiar electron-flow primitives into an unseen full mechanism composition.

## Structured failure certificates

`diagnose_proof` maps the first deterministic executor failure to a stable code and edge location, including:

- parse failures;
- missing atom maps;
- bond or lone-pair mismatches;
- charge precondition failures;
- electron non-conservation;
- invalid chemical states;
- unreachable edges;
- DAG join mismatches.

```python
from mechet import diagnose_proof, format_repair_feedback

certificate = diagnose_proof(prediction)
feedback = format_repair_feedback(certificate) if certificate else "OK"
```

This certificate can be returned to a model for local proof repair instead of regenerating the entire sequence.

## Deterministic local repair

Lone-pair lines certify an executed transition but do not mutate the molecular graph. When only these declarations are wrong, MechET can safely replace them with executor-derived values and re-run the proof.

```bash
python scripts/repair_mechet_proof_generations.py \
  --predictions outputs/mechet_proof_eval/generations.jsonl \
  --out outputs/mechet_proof_eval/generations.repaired.jsonl
```

Bond and charge failures are not silently repaired because changing those operations changes the executed chemistry. They are emitted as structured feedback for model-guided correction.

## Extended evaluation

```bash
python scripts/eval_mechet_proof_generations.py \
  --data data/mechet_proof_sft/valid.jsonl \
  --predictions outputs/mechet_proof_eval/generations.jsonl \
  --attempt-local-repair \
  --out outputs/mechet_proof_eval/summary.json
```

The report includes:

- execution before and after repair;
- endpoint exact match;
- partial-order proof equivalence;
- mechanism composition match;
- repair rate and failure codes;
- topology-stratified metrics.
