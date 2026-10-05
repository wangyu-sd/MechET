# mech-USPTO-31k full endpoint and external-baseline protocol

> **2026-10-05 target-field audit:** this frozen handoff is the historical
> `rxn_prod_min` **proxy**, not a verified desired-product benchmark. Selecting
> the largest organic component from the complete HF `rxn_prod_equ` field
> instead changes 6,430/24,959 train, 767/3,120 valid and 799/3,120 test
> targets. Typical old targets in the affected test rows are
> dicyclohexylurea (583) and isobutene (164), both common byproducts. Keep
> existing results for same-proxy lineage only; do not call their accuracy
> standard product-only retrosynthesis accuracy. The equ-field rule also
> remains a proxy pending comparison to the original reaction table. Audit
> code and per-reaction IDs are in PR #81, documented in
> `docs/SYSTEM_ONE_RETROSYNTHESIS.md`.

## Historical public-source min-field proxy

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
4. select its deterministic largest organic fragment as a target proxy; this
   can itself be a byproduct when the desired product is absent from
   `rxn_prod_min`;
5. map `initial species >> selected proxy product` once with RXNMapper 0.4.2 under
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
audit; it is now needed before claiming a verified desired-product target.
Its numeric atom-map labels
would not be model features because this protocol reindexes from the product
anyway.

The builder accepts `--product-field rxn_prod_equ` for a **separate** full-size
equ-field proxy artifact. This does not modify the old min-field output and
requires explicit new output and LocalRetro directories. Run the source audit
and remap all three splits before training or evaluating on the alternative;
never mix min-field and equ-field train/test files.

PR #81 completed that separate build at
`data/mech_uspto_31k_full_endpoint_rxnmapper_equ_proxy_v1_20261005/` with
24,959/3,120/3,120 rows and RXNMapper 0.4.2/Transformers 4.57.1. The
newly mapped LocalRetro CSVs have their own directory under
`data/baselines/localretro_mech_uspto_31k_rxnmapper_equ_proxy_v1_20261005/`.
The equ-field selection is still explicitly labelled a proxy; its existence
does not certify desired-product labels or validate old min-field results.

The legacy `data/mech_uspto_31k_full_endpoint_sft/` copied unmapped HF endpoint
strings into fields named `product_mapped` and `precursor_mapped`. It is invalid
for LocalRetro and is permanently excluded from new training.
