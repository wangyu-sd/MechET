# Data

- `samples/valid_mini.jsonl` — tiny gold examples (linear / tree / DAG) for smoke tests.
- Full SFT JSONL is built from FlowER with:

```bash
python scripts/build_mechet_sft.py --flower-root /path/to/flower_new_dataset --out-dir data/mechet_sft
```

If you already built data inside the parent `reflow` repo, you can symlink:

```bash
ln -s /path/to/reflow/data/orbit_mech_et_sft data/mechet_sft
```
