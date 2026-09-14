# MechET data processing for Retro-MTGR

## Entry points

- `flower_full_data_processing.py` processes the frozen Flower Full splits.
- `mech_uspto_31k_data_processing.py` processes the frozen mech-USPTO-31K splits.
- `retro_mtgr_data_processing.py` is the shared conversion and validation core.

The two entry points do not modify the source JSONL files and do not create a
new split. Reaction class is always `0`; reaction-class features must remain
disabled.

```bash
cd /home/estar/pxy/mechet/baselines/Retro-MTGR-main
PY=/home/estar/anaconda3/envs/gnn/bin/python

$PY -u flower_full_data_processing.py --workers 16
$PY -u mech_uspto_31k_data_processing.py --workers 12
```

Use `--input-dir` and `--output-dir` to process an audit copy instead of the
default full data. For example:

```bash
$PY -u flower_full_data_processing.py \
  --input-dir data/MechET/flower_full_audit100 \
  --output-dir audits/mechet_retro_mtgr/flower_full/native_processing_audit100 \
  --workers 4 --progress-every 0
```

## Output contract

Each output directory contains:

- `train.txt`, `valid.txt`, and `test.txt`: the official five-column native
  format (`class`, mapped product, bond indices, mapped R1, mapped R2).
- `*_index.jsonl`: one row per source case, including stable ID, explicit
  support status/reason, and the corresponding native row index when present.
- `*_targets.jsonl`: one row per source case with the strict adapter target.
- `label_vocabulary.json`: labels derived from supported training rows only.
- `manifest.json`: inputs, hashes, row counts, support reasons, and policies.
- `validation_report.json`: machine-readable invariant checks.

Retro-MTGR's native target can express only two-fragment precursors produced by
one product-bond deletion with local endpoint leaving groups. Therefore the
native `.txt` files contain supported cases only. The index and target JSONL
files retain all cases in original order. During validation/test evaluation,
cases without a native prediction must be included by stable ID and scored as
failures; evaluating only `valid.txt` or `test.txt` would inflate results.

Unlike the original scripts, the converter relocates the center from atom-map
numbers after canonical product serialization. This guarantees the recorded
atom indices identify a real bond in the exact product SMILES written to the
native file. R1 and R2 are then ordered by the maps at those two endpoints.

## Verified full-data results

| Dataset | Split | Source/retained rows | Native supported rows | Supported rate |
|---|---:|---:|---:|---:|
| Flower Full | train | 257,171 | 190,008 | 73.884% |
| Flower Full | valid | 2,890 | 2,117 | 73.253% |
| Flower Full | test | 28,971 | 21,384 | 73.812% |
| mech-USPTO-31K | train | 24,959 | 13,193 | 52.859% |
| mech-USPTO-31K | valid | 3,120 | 1,672 | 53.590% |
| mech-USPTO-31K | test | 3,120 | 1,614 | 51.731% |

The full runs passed every check in their `validation_report.json` files:

- source, target, and index row counts match;
- source stable-ID order and source product/precursor fields are preserved;
- stable IDs are unique within splits and disjoint across splits;
- every native row has five columns, class `0`, parseable structures, a real
  product center bond, and endpoint maps aligned with R1/R2;
- native row counts equal the number of explicitly supported targets;
- the label vocabulary exactly equals labels observed in supported train rows.

Full outputs are in:

- `data/MechET/flower_full_retro_mtgr_processed`
- `data/MechET/mech_uspto_31k_full_retro_mtgr_processed`

The verified train-only label vocabulary sizes are 1,140 for Flower Full and
444 for mech-USPTO-31K. The held-out sets contain unseen label keys; these are
reported in the manifests and must not be added to the training vocabulary.
