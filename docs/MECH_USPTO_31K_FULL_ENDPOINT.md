# mech-USPTO-31k full endpoint and LocalRetro mapping contract

## Critical source distinction

The Hugging Face elementary-step parquet is **not** the reaction atom-mapping
source. In that export, `rxn_prod_min`, `rxn_prod_equ`, `elem_reac_*`, and
`elem_prod_*` are unmapped. `mech_smi_*` maps only atoms referenced by one
elementary arrow, so those local identifiers cannot be copied into a complete
reaction-level LocalRetro example.

The full endpoint must instead be derived from the original Figshare v2
reaction-level table and its mapped `reaction` column. The Hugging Face export
supplies the frozen `source`-to-split assignment and `rxn_prod_min` only as a
join audit: after maps are removed, the original desired product must occur in
the mechanism final mixture.

An artifact for which `product_mapped == product_unmapped` is invalid. Do not
train LocalRetro, GLN, RetroXpert, RetroComposer, or NeuralSym from it.

## Required mapping invariants

For every reaction:

1. every product atom has one unique positive map;
2. every product map occurs exactly once on the reactant side;
3. original map integers are discarded;
4. product atoms are reassigned contiguous maps from product-only canonical
   ranks;
5. the same permutation is transported to the reactants;
6. reactant-only atoms receive fresh identifiers above the product range;
7. removing maps recovers the original desired product and precursor
   structures;
8. no reaction may be silently filtered from 24,959 / 3,120 / 3,120.

LocalRetro receives the synchronized `reactants>reagents>products` reaction,
not a separately numbered product. The required CSV `class` column is a
constant `0` placeholder and reaction-class features are disabled in the
default matched condition.

## Build

Download the original Figshare v2 CSV, then run:

```bash
python scripts/build_mech_uspto31k_full_endpoint_sft.py \
  --mapped-reactions /absolute/path/to/figshare_v2.csv \
  --mapped-id-column 'rxn id' \
  --mapped-reaction-column reaction \
  --hf-root data/raw/mech_uspto_31k/data \
  --output-dir data/mech_uspto_31k_full_endpoint_sft \
  --localretro-dir data/baselines/localretro_mech_uspto_31k
```

The command is fail-closed. A missing join, incomplete product mapping,
cross-side map inconsistency, product/reference mismatch, duplicate ID, or
split-count mismatch aborts the build rather than reducing the benchmark.
