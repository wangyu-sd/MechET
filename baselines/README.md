# External baseline source snapshots

This directory contains runnable source/configuration snapshots for the nine
external retrosynthesis baselines used by MechET. It intentionally does not
contain reaction datasets, generated preprocessing artifacts, experiment
outputs, virtual environments, or trained model weights.

The source snapshots are copied rather than moved from the sibling
`../baselines/` directory. Existing files in this directory take precedence so
that local MechET adaptations are not overwritten. Re-run
`scripts/sync_baseline_sources.sh` to copy newly added source files.

## Included baselines

- `LocalRetro-main`
- `ReactSeq-main`
- `Retro-MTGR-main`
- `RetroBridge-main`
- `RetroDFM-R-main`
- `RetroSynFlow-main`
- `RxnNano-main`
- `editretro-master`
- `retrosynthesis-main` (R-SMILES)

Vendored implementation trees such as ReactSeq's `onmt`, EditRetro's
`fairseq`, and RetroDFM-R's `slime` are retained. Generated environments and
compiled extensions are not; install each baseline's declared dependencies in
an external environment.

## Raw-data contract

Supply raw data locally at the paths below, or bind/symlink those paths to
storage outside the Git checkout. Generated preprocessing and checkpoints may
use the same directories, but all such artifacts are ignored by the parent
repository.

| Baseline | Expected local input layout | In-repository preparation entry point |
|---|---|---|
| LocalRetro | `data/<dataset>/raw_train.csv`, `raw_val.csv`, `raw_test.csv` | `preprocessing/Extract_from_train_data.py`, `preprocessing/Run_preprocessing.py` |
| ReactSeq | MechET `train.jsonl`, `valid.jsonl`, `test.jsonl` supplied with `--input` | `prepare_mechet_reactseq.py prepare` |
| Retro-MTGR | `data/MechET/<dataset>/{train,valid,test}.jsonl` | `retro_mtgr_data_processing.py` |
| RetroBridge | MechET JSONL split directory supplied with `--source-dir` | `prepare_mechet_retrobridge.py` |
| RetroDFM-R | `data/pretrain/train.jsonl`, `data/cold_start/train.jsonl`, RL data, or `data/eval/<dataset>/*.jsonl` | upstream `swift_scripts/` and `slime/scripts/retrodfm/` launchers |
| RetroSynFlow | `data/<dataset>/raw/uspto50k_{train,val,test}.csv` | `scripts/preprocess_mechet.py` |
| RxnNano | `data/train_<name>.jsonl`, `validation_<name>.jsonl`, `test_<name>.jsonl` | `src/data/` formatters/loaders and `audit_mechet_rxnnano.py preprocess` |
| EditRetro | `datasets/<dataset>/raw/raw_train.csv`, `raw_val.csv`, `raw_test.csv` | `scripts/run_mechet_preprocess.sh` |
| R-SMILES | `dataset/<dataset>/raw_train.csv`, `raw_val.csv`, `raw_test.csv` | `preprocessing/generate_PtoR_data.py` |

MechET's shared exporters are under `scripts/`, notably
`prepare_collaborator_b_baseline_data.py`, `export_localretro_raw_csv.py`, and
`export_rxnnano_jsonl.py`. They can populate these local layouts from the
frozen external-baseline handoff without committing the resulting data.

## Repository policy

The root `.gitignore` rejects common reaction-data formats, generated dataset
directories, checkpoints, model weights, caches, logs, and environments below
`baselines/`. The intentionally retained small support artifacts are:

- LocalRetro's `data/configs/default_config.json`;
- ReactSeq's `datasets/vocabs/whole_vocabs.src`;
- Retro-MTGR's elemental and bond metadata in
  `data/USPT-50K/other_data/`.

The destination snapshot also normalizes CRLF shell-script line endings. In
the bundled EditRetro/Fairseq tree, three flattened symlink placeholder files
are represented as Python compatibility shims, and an unmatched parenthesis
in the unused `pointernet.py` demonstration block is repaired. The sibling
source tree remains unchanged.

Before committing, validate the required source files and source-only policy:

```bash
python scripts/check_baseline_snapshot.py
```

For a quick suffix-only check, run:

```bash
find baselines -type f \
  \( -name '*.csv' -o -name '*.jsonl' -o -name '*.pt' -o -name '*.pth' \
     -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.bin' \)
```

The command should print no files.
