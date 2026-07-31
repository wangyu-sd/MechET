# Data lineage, leakage audit, and benchmark-freezing protocol

> **Status: current companion protocol.** This document is authoritative for dataset lineage and decontamination only. Model training, losses, inference, required results, and paper claims are defined in [`PROOF_CENTRIC_EXPERIMENT_PLAN.md`](PROOF_CENTRIC_EXPERIMENT_PLAN.md).

## 1. Required source data

1. FlowER-derived MechET rows with stable IDs, mapped products, proof/state targets, and source split metadata.
2. USPTO-50K standard train/valid/test reaction tables with stable IDs and mapped reaction SMILES.
3. USPTO-MIT and USPTO-FULL for secondary source-corpus audits.
4. Patent ID, publication date, and patent-family metadata when recoverable.
5. External non-USPTO or post-cutoff mechanisms for later NMI-stage validation.

If patent metadata is absent, patent-family and temporal disjointness must be reported as unverifiable. Chemical canonicalization cannot replace patent-family auditing.

## 2. Benchmark-first policy

The required order is:

```text
freeze benchmark bytes and hashes
 -> freeze normalization configuration
 -> compute train–benchmark overlap
 -> quarantine conflicting training rows
 -> rebuild training manifests
 -> train models
```

The test set is never filtered after model selection. Every removed training row is written to a quarantine JSONL containing its stable ID, conflict reasons, and chemical keys.

## 3. Frozen overlap levels

### `exact_full`

Canonicalized full reactant, reagent, and product multiset.

### `exact_structural`

Canonicalized atom-contributing precursor fragments plus product. Free solvents, catalysts, salts, and spectators are excluded.

### `product`

Exact canonical product identity.

### `scaffold`

Product Bemis–Murcko scaffold.

### `reaction_center`

Changed-bond and local changed-atom-role signature.

### `proof_composition`

Map-label- and serialization-invariant elementary proof composition.

### `patent`

Patent or patent-family identifier when retained by both corpora.

The audit may additionally report maximum product fingerprint similarity, but similarity thresholds must not replace exact overlap counts.

## 4. Required commands

```bash
python scripts/audit_reaction_overlap.py \
  --train data/mechet_proof_sft/train.jsonl \
  --benchmark data/benchmarks/uspto50k/test.csv \
  --benchmark-format reaction_table \
  --reaction-field reaction_smiles \
  --out-dir outputs/data_audit/flower_vs_uspto50k_test
```

Repeat for:

```text
USPTO-50K train
USPTO-50K valid
USPTO-50K test
USPTO-MIT test
USPTO-FULL test
```

Build clean conditions by removing conflicts from training:

```bash
python scripts/build_decontaminated_dataset.py \
  --train data/mechet_proof_sft/train.jsonl \
  --benchmark data/benchmarks/uspto50k/test.csv \
  --output data/mechet_proof_clean/train.jsonl \
  --manifest data/mechet_proof_clean/manifest.json \
  --policy exact_structural product
```

Minimum training conditions:

```text
exact-clean: exact_structural + product
scaffold-clean: exact_structural + product + scaffold
center-clean: exact_structural + product + scaffold + reaction_center
```

## 5. Structural precursor endpoint

A reactant fragment is structural when it shares at least one atom map with the product. The complete fragment is retained, including leaving-group atoms that do not appear in the product.

Free fragments with no product-contributing atom maps are treated as environment species and are not part of the primary endpoint score.

Unmapped rows cannot support this distinction and must be flagged rather than silently classified.

## 6. Atom-map leakage control

Atom maps are nuisance labels. A valid control must consistently permute every map appearing in:

```text
product SMILES
root and edge imports
BOND actions
LP actions
CHARGE actions
full precursor
structural precursor
```

```bash
python scripts/build_map_permutations.py ...
```

Report the performance drop between canonical and multiple random map permutations. Randomly changing only product maps is not a valid control.

## 7. Required audit outputs

Every official audit directory must contain:

```text
input file paths and SHA-256 hashes
row counts
field names
source/version information
normalization configuration and digest
overlap counts and rates at every level
per-row conflict records
quarantine file
retained and removed train sizes
patent/date coverage
product-similarity distribution where computed
```

## 8. Required paper table

Rows:

```text
FlowER train -> USPTO-50K train
FlowER train -> USPTO-50K valid
FlowER train -> USPTO-50K test
FlowER train -> USPTO-MIT test
FlowER train -> USPTO-FULL test
```

Columns:

```text
exact_full
exact_structural
product
scaffold
reaction_center
proof_composition
patent
```

A second table must report original training size, removed rows, retained rows, and removal-reason distribution for exact-, scaffold-, and center-clean conditions.

## 9. Interpretation rules

- Standard USPTO results support literature comparability, not independent external validation.
- Exact-clean results reduce direct identity leakage but do not establish scaffold or template independence.
- Scaffold-clean results do not establish reaction-center independence.
- Reaction-center-clean results still may share broader mechanism primitives.
- MechComp-OOD tests composition generalization and must be reported separately from source-corpus leakage.
- A foundation model may have unknown pretraining exposure to public USPTO files; matched-backbone comparisons and from-scratch controls should be used where feasible.

## 10. Training gate

Do not start headline training until:

1. benchmark hashes and normalization digests are frozen;
2. overlap matrices are generated;
3. quarantine manifests are generated;
4. proof and state corpora are matched by stable ID;
5. structural endpoints agree across all baseline task variants.

Continue with the authoritative experiment plan after this gate passes.
