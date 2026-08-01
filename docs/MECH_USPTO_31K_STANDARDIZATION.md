# mech-USPTO-31k standardization

`mech_uspto_31k` stores atom maps primarily in `mech_smi_*`, and those strings map only atoms referenced by the current arrow code. The accompanying `elem_reac_*` and `elem_prod_*` fields are usually unmapped.

The standardizer therefore uses the following source-specific path:

1. parse the current state and arrow pairs from `mech_smi_*`;
2. preserve the explicit reactive atom maps;
3. deterministically assign unique maps to previously unmapped atoms;
4. retain explicit mapped hydrogens during parsing;
5. execute the coupled electron moves with the formal step executor;
6. use the executor-derived state as the mapped target product;
7. compare reactant and product heavy-atom graphs against the associated unmapped `elem_reac_*` and `elem_prod_*` fields;
8. quarantine rows whose moves cannot execute or whose reference structures disagree.

The ordinary command remains:

```bash
python scripts/forward_expert_data.py standardize \
  --input data/raw/mech_uspto_31k \
  --output data/forward_expert/reactions.jsonl \
  --source mech_uspto_31k

python scripts/forward_expert_data.py build \
  --input data/forward_expert/reactions.jsonl \
  --output-dir data/forward_expert/steps
```

No RXNMapper pass is required for these elementary-step rows. The completed maps are local deterministic identifiers for model input; the original arrow-referenced maps are never changed. The standardized metadata records the chosen mechanistic variant, assigned maps, reference structures, audit results, and that the mapped target was produced by formal electron-step execution.

Rows without a valid `mech_smi_*` arrow sequence continue through the generic mapped-reaction path. `--allow-unmapped` is intended only for outcome-only compatibility data and does not create source/sink supervision.
