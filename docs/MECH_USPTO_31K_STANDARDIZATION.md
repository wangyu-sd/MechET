# mech-USPTO-31k standardization

`mech_uspto_31k` stores atom maps primarily in `mech_smi_*`; those strings map only atoms referenced by the current arrow code. The accompanying `elem_reac_*` and `elem_prod_*` fields are usually unmapped.

The standardizer parses the mapped reactive state and arrows from `mech_smi_*`, deterministically fills maps for previously unmapped atoms, preserves explicit mapped hydrogens, executes the coupled moves, and uses the executor-derived mapped state as the target. It then audits heavy-atom reactant and product structures against the corresponding unmapped `elem_reac_*` and `elem_prod_*` fields. Inconsistent rows are quarantined.

```bash
python scripts/forward_expert_data.py standardize \
  --input data/raw/mech_uspto_31k \
  --output data/forward_expert/reactions.jsonl \
  --source mech_uspto_31k

python scripts/forward_expert_data.py build \
  --input data/forward_expert/reactions.jsonl \
  --output-dir data/forward_expert/steps
```

RXNMapper is not required for these elementary-step rows. Original arrow-referenced map labels are retained; newly assigned maps are local deterministic identifiers recorded in metadata.
