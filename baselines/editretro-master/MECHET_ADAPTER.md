# EditRetro on the frozen MechET endpoint benchmarks

This adapter runs the published EditRetro representation and model on the two
complete MechET reaction-level splits. It does not use executable traces,
executor states, or a generic autoregressive product-to-reactant replacement.

## Preserved native method

- mapped endpoints are converted to root-aligned, atom-map-free SMILES;
- the published `SPE_ChEMBL.txt` tokenizer and all `dict.txt` token IDs are
  retained; missing SPE tokens are appended after them from the complete
  training split only;
- the pretraining token-decoder stage is followed by EditRetro fine-tuning;
- `editretro_nat` dynamically derives reposition and insertion oracle labels;
- inference starts from the product and uses 10 iterative edit steps;
- 20 native hypotheses are formed as `repos_beam=5`, `mask_beam=1`, and
  `token_beam=4`;
- ten augmented product views are fused with the published reciprocal-rank
  score (`score_alpha=0.1`).

Two completeness guards intentionally replace silent filters in the official
preprocessor:

- valid small-molecule reactions are retained (the official `<5`-reactant-atom
  heuristic would remove one mech-USPTO training reaction);
- the common MechET handoff has already removed auxiliary reagents, so a
  structural precursor fragment with no shared product atom map is retained as
  a canonical, unrooted suffix. This affects four FlowER train rows and one
  FlowER test row; every other fragment follows the official root alignment.

These guards change neither the model nor its oracle edit construction. They
prevent five mapping anomalies and one valid small reaction from being silently
removed from the frozen benchmark.

## Environment

Use a GPU node with CUDA 11.6 and GCC 9.4, as required by the bundled Fairseq
CUDA extension.

```bash
cd /home/estar/pxy/mechet/baselines/editretro-master
python3.10 -m venv .venv-editretro
source .venv-editretro/bin/activate
python -m pip install --upgrade pip ninja
python -m pip install \
  torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 \
  --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r requirements-mechet.txt
python -m pip install --editable ./fairseq
python -m pip freeze > environment.mechet.lock.txt
```

Confirm that `fairseq/fairseq/libnat_cuda*.so` exists before preprocessing or
training.

## 1. Mandatory 100-reaction audit

The raw CSV and stable-ID indexes are already present. Run native SPE
preprocessing, Fairseq binarization, and the identity/OOV/coverage audit:

```bash
OVERWRITE=1 ./scripts/run_mechet_preprocess.sh mech_uspto_31k_full audit100
OVERWRITE=1 ./scripts/run_mechet_preprocess.sh flower_full audit100
```

Each validated directory contains `ARTIFACT_STATUS.json` with
`training_allowed: true`. Training scripts refuse an unvalidated artifact.

The dictionary audit treats any train OOV or held-out **source** OOV as a hard
error. A held-out target-only OOV is reported but is never copied into the
training vocabulary. FlowER test has exactly one such reaction (`[Ar]`, ten
augmented target lines); it remains in the denominator.

## 2. Small overfit test

The `audit100` profile is within the required 32--128 reaction range. The
following values are intentionally short smoke/overfit settings; retain all
model objectives and decoding settings.

```bash
RUN_NAME=overfit MAX_UPDATE=500 MAX_EPOCH=100 \
SAVE_INTERVAL_UPDATES=100 WARMUP_UPDATES=50 MAX_TOKENS=4096 \
./scripts/run_mechet_pretrain.sh mech_uspto_31k_full audit100

AVERAGE_COUNT=5 ./scripts/prepare_mechet_pretrain_checkpoint.sh \
  results/mechet/mech_uspto_31k_full/audit100/pretrain/overfit/checkpoints

RUN_NAME=overfit MAX_UPDATE=1000 MAX_EPOCH=200 \
SAVE_INTERVAL_UPDATES=200 WARMUP_UPDATES=50 MAX_TOKENS=4096 \
./scripts/run_mechet_finetune.sh mech_uspto_31k_full \
  results/mechet/mech_uspto_31k_full/audit100/pretrain/overfit/checkpoints/pretrain_for_finetune.pt \
  audit100

AVERAGE_COUNT=5 ./scripts/average_mechet_finetune_checkpoints.sh \
  results/mechet/mech_uspto_31k_full/audit100/finetune/overfit/checkpoints

./scripts/run_mechet_generate.sh mech_uspto_31k_full \
  results/mechet/mech_uspto_31k_full/audit100/finetune/overfit/checkpoints/finetune_average.pt \
  audit100 train
```

Check that the losses decrease, generated candidates are valid, and every
prediction maps back to the original stable ID.

### Completed CPU audit run

The host had no usable NVIDIA driver, so the same `audit100` recipe was also
validated with the official CPU `libnat` extension.  CPU runs require
`NUM_WORKERS=0` in restricted environments, and the chemistry export can use a
separate Python environment containing RDKit:

```bash
CPU=1 NUM_WORKERS=0 NO_EPOCH_CHECKPOINTS=1 \
PYTHON_BIN=/home/estar/anaconda3/envs/pxy/bin/python \
RUN_NAME=overfit_cpu MAX_UPDATE=500 MAX_EPOCH=100 \
SAVE_INTERVAL_UPDATES=100 WARMUP_UPDATES=50 MAX_TOKENS=512 \
./scripts/run_mechet_pretrain.sh mech_uspto_31k_full audit100

# Convert the averaged pretraining checkpoint, then fine-tune as above with
# MAX_UPDATE=1000 and SAVE_INTERVAL_UPDATES=200.

CPU=1 NUM_WORKERS=0 \
PYTHON_BIN=/home/estar/anaconda3/envs/pxy/bin/python \
CHEM_PYTHON_BIN=/home/estar/anaconda3/envs/mechet/bin/python \
./scripts/run_mechet_generate.sh mech_uspto_31k_full \
  results/mechet/mech_uspto_31k_full/audit100/finetune/overfit_cpu/checkpoints/finetune_average.pt \
  audit100 train
```

The completed run produced the following audit evidence.  These numbers are
pipeline/overfit diagnostics only and are not headline benchmark scores.

| Check | Result |
|---|---:|
| pretraining updates | 500 |
| pretraining train loss, first epoch / final partial epoch | 10.804 / 8.885 |
| fine-tuning updates | 1,000 |
| fine-tuning train loss, first epoch / final partial epoch | 19.190 / 8.421 |
| original reactions / augmented views / native hypotheses | 100 / 1,000 / 20,000 |
| rows with at least one valid candidate | 89 / 100 |
| valid unified Top-10 candidate slots | 320 / 1,000 |
| missing views / incomplete native beams | 0 / 0 |
| stable IDs in exact frozen-reference order | 100 / 100 |
| CPU inference time | 1,123.2 ms per original reaction |

The unified predictions and reports are under
`results/mechet/mech_uspto_31k_full/audit100/generation/finetune_average/train/`.
The shared evaluator is invoked with `candidate_semantics=native_ranked`, so
EditRetro prefixes are reported as Success@K rather than stochastic Pass@K.

## 3. Full matched runs

Pretraining is dataset-specific and uses only that frozen training split,
avoiding hidden overlap from the public USPTO pretrained checkpoint. The
resumable end-to-end entry point runs preprocessing/binarization, pretraining,
checkpoint averaging, iterative-edit finetuning, final checkpoint averaging,
native decoding/ranking, unified export, and MechET evaluation:

```bash
PIPELINE_OUTPUT_ROOT=/data/pxy/models/EditRetro \
  ./scripts/run_mechet_full_pipeline.sh mech_uspto_31k_full 4 official_full_seed1

PIPELINE_OUTPUT_ROOT=/data/pxy/models/EditRetro \
  ./scripts/run_mechet_full_pipeline.sh flower_full 5 official_full_seed1
```

The launcher resumes pretraining from `checkpoint_last.pt`, resumes finetuning
from its own `checkpoint_last.pt`, and skips completed averaged or evaluation
artifacts. It retains the last five pretraining and last ten finetuning update
checkpoints, matching the averaging windows without unbounded disk growth.

Full test output is written directly to:

```text
/home/estar/pxy/mechet/MechET/outputs/external_baselines/editretro/
  mech_uspto_31k_full/predictions.jsonl
  flower_full/predictions.jsonl
```

Every test reaction remains in the denominator. Rows with missing views or no
valid native prediction receive ten empty candidates and a non-`ok`
`evaluation_status`; they are never removed before MechET evaluation.

## Current workspace verification

The complete native text preprocessing has already been run in this workspace:

| Dataset | train views | valid views | test views | failures | identity/OOV result |
|---|---:|---:|---:|---:|---|
| mech-USPTO-31k full | 249,590 | 31,200 | 31,200 | 0 | pass; no OOV |
| FlowER full | 2,571,710 | 28,900 | 289,710 | 0 | pass; one disclosed target-only `[Ar]` OOV |

The reports are in `datasets/<dataset>/aug10/mechet_preprocessing_audit.json`.
Both full data-bins were completed and re-audited on 2026-09-07, and both
artifact gates now record `status: validated` and `training_allowed: true`.
The two `official_full_seed1` pretraining runs are active in tmux sessions
`pxy` (mech-USPTO, GPU 4) and `pxy1` (FlowER, GPU 5); checkpoints and pipeline
logs are under `/data/pxy/models/EditRetro/<dataset>/full/official_full_seed1/`.
