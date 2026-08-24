# mech-USPTO-31k full endpoint and external-baseline protocol

## Active public-source protocol

The active build uses the public Hugging Face snapshot
`SchwallerGroup/mech_uspto_31k`, rather than waiting for Figshare. The frozen
reaction-level split contains 24,959 train, 3,120 validation, and 3,120 test
reactions.

The HF files store elementary steps, so one reaction pair is reconstructed as:

1. group rows by `rxn_idx` under the upstream split;
2. take `elem_reac_spe` at `step_idx_forward == 0` as the complete initial
   species mixture (`elem_reac_min` is only the current elementary-step input
   and is therefore insufficient for multi-step reactions);
3. canonicalize `rxn_prod_min`, which is invariant within the reaction;
4. select its deterministic largest organic fragment as the desired-product
   proxy, excluding salts and smaller byproducts from the retrosynthesis target;
5. map `initial species >> desired product` once with RXNMapper 0.4.2 under
   Transformers 4.57.1;
6. discard RXNMapper's numeric map labels and apply product-only canonical
   reindexing, transporting the same map permutation to the precursor side;
7. export one shared mapping for every external method. Downstream repositories
   must not independently remap or resplit the data.

This build is fail-closed: a missing reaction, invalid structure, mapping that
changes either unmapped endpoint, incomplete product-map transport, duplicate
ID, or split-count mismatch aborts the build. Proof compilation and executor
replay are not filtering criteria.

## Build

Download the three public HF parquet shards (or reuse the byte-identical frozen
copies already under `data/raw/mech_uspto_31k/data/`):

```bash
mkdir -p data/raw/mech_uspto_31k/data
for split in train val test; do
  curl -L --fail --retry 5 \
    "https://huggingface.co/datasets/SchwallerGroup/mech_uspto_31k/resolve/main/data/${split}-00000-of-00001.parquet" \
    -o "data/raw/mech_uspto_31k/data/${split}-00000-of-00001.parquet"
done
sha256sum data/raw/mech_uspto_31k/data/*.parquet
```

Expected SHA-256 values are `58852eeaac5a479d914ed674451f441485e3a14f7ed0b76841bfc38882ac1eed`
(train), `baa11fc1cad03fbdc353f35dbbf44832cb79001e786051f8d1cd294b0fdc67cf`
(validation), and `389a5d70dfe7aeb436a0d764e8c8a054e97654dc63fdb91eda535925393da611`
(test).

Then build the shared mapping:

```bash
python -m venv .venv-rxnmapper
.venv-rxnmapper/bin/pip install -r requirements/rxnmapper.txt

.venv-rxnmapper/bin/python scripts/build_mech_uspto31k_rxnmapper_baseline.py \
  --hf-root data/raw/mech_uspto_31k/data \
  --output-dir data/mech_uspto_31k_full_endpoint_rxnmapper \
  --localretro-dir data/baselines/localretro_mech_uspto_31k_rxnmapper
```

The mapping environment is intentionally separate from the Qwen training
environment because the frozen RXNMapper release uses Transformers 4.x.

Then freeze the common method-agnostic handoff:

```bash
python scripts/export_full_baseline_pairs.py \
  --datasets mech_uspto_31k_full \
  --mech-uspto-dir data/mech_uspto_31k_full_endpoint_rxnmapper \
  --output-root data/external_baselines
```

Expected outputs:

```text
data/mech_uspto_31k_full_endpoint_rxnmapper/{train,valid,test}.jsonl
data/baselines/localretro_mech_uspto_31k_rxnmapper/{train,valid,test}.csv
data/external_baselines/mech_uspto_31k_full/{train,valid,test}.jsonl
```

All manifests record source hashes, output hashes, stable-ID hashes, mapping
confidence summaries, mapping versions, and zero executor filtering.

## External-method contract

- LocalRetro starts from the shared mapped CSV and extracts templates from
  train only. Its `class` column is the constant `0`, with reaction-class
  features disabled.
- ReactSeq and EditRetro derive their official mapped operations from the same
  frozen reaction mapping.
- R-SMILES starts from the unmapped product/precursor fields and applies only
  its published root alignment.
- RetroBridge and other graph methods derive graph pairs from the same stable
  IDs.
- Every method predicts all 3,120 test IDs and preserves `stable_id`; missing
  predictions count as failures.

## Figshare relation and invalid legacy artifact

The original Figshare v2 `reaction` table remains useful as a future provenance
audit, but it is not required by the active build. Its numeric atom-map labels
would not be model features because this protocol reindexes from the product
anyway.

The legacy `data/mech_uspto_31k_full_endpoint_sft/` copied unmapped HF endpoint
strings into fields named `product_mapped` and `precursor_mapped`. It is invalid
for LocalRetro and is permanently excluded from new training.
