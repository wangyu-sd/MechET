#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 {mech_uspto_31k_full|flower_full} GPU[,GPU...] [RUN_NAME]" >&2
    exit 2
fi

dataset=$1
gpus=$2
run_name=${3:-official_full_seed1}
case "$dataset" in
    mech_uspto_31k_full|flower_full) ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac
if [[ ! "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPUs must be physical CUDA device indices separated by commas: $gpus" >&2
    exit 2
fi

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-/home/estar/anaconda3/envs/pxy/bin/python}
chem_python_bin=${CHEM_PYTHON_BIN:-/home/estar/anaconda3/envs/mechet/bin/python}
mechet_dir=${MECHET_DIR:-/home/estar/pxy/mechet/MechET}
output_root=${PIPELINE_OUTPUT_ROOT:-/data/pxy/models/EditRetro}
run_dir="$output_root/$dataset/full/$run_name"
pretrain_dir="$run_dir/pretrain/checkpoints"
finetune_dir="$run_dir/finetune/checkpoints"
generation_dir="$run_dir/generation/test"
artifact_dir="$repo_dir/datasets/$dataset/aug10"
mkdir -p "$pretrain_dir" "$finetune_dir" "$generation_dir"

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
stage() { echo "[$(timestamp)] $*"; }

stage "dataset=$dataset gpus=$gpus run=$run_name output=$run_dir"

if "$python_bin" "$repo_dir/utils/require_validated_artifact.py" "$artifact_dir" >/dev/null 2>&1; then
    stage "validated data-bin already exists; preprocessing skipped"
else
    stage "preprocessing and binarizing complete reaction-level data"
    reuse_preprocessed=0
    if [[ -s "$artifact_dir/preprocess_manifest.json" && \
          -s "$artifact_dir/mechet_preprocessing_audit.json" && \
          -s "$artifact_dir/dict.mechet.txt" ]]; then
        reuse_preprocessed=1
    fi
    PYTHON_BIN="$python_bin" CHEM_PYTHON_BIN="$chem_python_bin" \
        MECHET_DIR="$mechet_dir" AUGMENTATION=10 \
        REUSE_PREPROCESSED="$reuse_preprocessed" \
        PROCESSES=${PROCESSES:--1} BINARIZE_WORKERS=${BINARIZE_WORKERS:-20} \
        "$repo_dir/scripts/run_mechet_preprocess.sh" "$dataset" full
fi
"$python_bin" "$repo_dir/utils/require_validated_artifact.py" "$artifact_dir"

pretrain_checkpoint="$pretrain_dir/pretrain_for_finetune.pt"
if [[ -s "$pretrain_checkpoint" ]]; then
    stage "prepared pretraining checkpoint already exists; pretraining skipped"
else
    pretrain_save_interval=${PRETRAIN_SAVE_INTERVAL_UPDATES:-10000}
    if [[ "$dataset" == "mech_uspto_31k_full" && -z "${PRETRAIN_SAVE_INTERVAL_UPDATES:-}" ]]; then
        # This split has about 78 updates/epoch, so the official 300 epochs
        # finish before five 10k-spaced update checkpoints exist.
        pretrain_save_interval=4000
    fi
    stage "starting official EditRetro pretraining"
    PYTHON_BIN="$python_bin" GPUS="$gpus" RUN_NAME="$run_name" CHECKPOINT_DIR="$pretrain_dir" \
        SAVE_INTERVAL_UPDATES="$pretrain_save_interval" \
        NO_EPOCH_CHECKPOINTS=1 KEEP_INTERVAL_UPDATES=${PRETRAIN_AVERAGE_COUNT:-5} \
        "$repo_dir/scripts/run_mechet_pretrain.sh" "$dataset" full
    stage "averaging the last ${PRETRAIN_AVERAGE_COUNT:-5} pretraining update checkpoints"
    PYTHON_BIN="$python_bin" AVERAGE_COUNT=${PRETRAIN_AVERAGE_COUNT:-5} \
        "$repo_dir/scripts/prepare_mechet_pretrain_checkpoint.sh" \
        "$pretrain_dir" "$pretrain_checkpoint"
fi

finetune_checkpoint="$finetune_dir/finetune_average.pt"
if [[ -s "$finetune_checkpoint" ]]; then
    stage "averaged finetuning checkpoint already exists; finetuning skipped"
else
    # The upstream 2,000-epoch recipe targets a much smaller update count per
    # epoch. Flower has 2,571,710 augmented training examples and is trained on
    # two GPUs here (about 435 updates/epoch), so leave enough updates for the
    # 100-epoch cap to be reached.
    finetune_max_epoch_default=2000
    finetune_max_update_default=2000000
    finetune_warmup_updates_default=10000
    finetune_save_interval_updates_default=10000
    finetune_average_count_default=10
    if [[ "$dataset" == "flower_full" ]]; then
        finetune_max_epoch_default=100
        finetune_max_update_default=50000
        finetune_warmup_updates_default=3000
        finetune_save_interval_updates_default=2000
        finetune_average_count_default=5
    fi
    finetune_max_epoch=${FINETUNE_MAX_EPOCH:-${MAX_EPOCH:-$finetune_max_epoch_default}}
    finetune_max_update=${FINETUNE_MAX_UPDATE:-${MAX_UPDATE:-$finetune_max_update_default}}
    finetune_warmup_updates=${FINETUNE_WARMUP_UPDATES:-${WARMUP_UPDATES:-$finetune_warmup_updates_default}}
    finetune_save_interval_updates=${FINETUNE_SAVE_INTERVAL_UPDATES:-${SAVE_INTERVAL_UPDATES:-$finetune_save_interval_updates_default}}
    finetune_average_count=${FINETUNE_AVERAGE_COUNT:-$finetune_average_count_default}
    stage "starting official iterative-edit finetuning"
    stage "finetuning schedule: max_epoch=$finetune_max_epoch max_update=$finetune_max_update warmup_updates=$finetune_warmup_updates save_interval_updates=$finetune_save_interval_updates"
    finetune_resume=()
    if [[ -s "$finetune_dir/checkpoint_last.pt" ]]; then
        stage "resuming finetuning from checkpoint_last.pt"
        finetune_resume=(RESUME_CHECKPOINT="$finetune_dir/checkpoint_last.pt")
    fi
    env PYTHON_BIN="$python_bin" GPUS="$gpus" RUN_NAME="$run_name" CHECKPOINT_DIR="$finetune_dir" \
        MAX_TOKENS="${FINETUNE_MAX_TOKENS:-40960}" \
        MAX_EPOCH="$finetune_max_epoch" MAX_UPDATE="$finetune_max_update" \
        WARMUP_UPDATES="$finetune_warmup_updates" SAVE_INTERVAL_UPDATES="$finetune_save_interval_updates" \
        NO_EPOCH_CHECKPOINTS=1 KEEP_INTERVAL_UPDATES="$finetune_average_count" \
        "${finetune_resume[@]}" \
        "$repo_dir/scripts/run_mechet_finetune.sh" \
        "$dataset" "$pretrain_checkpoint" full
    stage "averaging the last $finetune_average_count finetuning update checkpoints"
    PYTHON_BIN="$python_bin" AVERAGE_COUNT="$finetune_average_count" \
        "$repo_dir/scripts/average_mechet_finetune_checkpoints.sh" \
        "$finetune_dir" "$finetune_checkpoint"
fi

prediction_file="$mechet_dir/outputs/external_baselines/editretro/$dataset/predictions.jsonl"
evaluation_file="${prediction_file%.jsonl}.evaluation.json"
if [[ -s "$prediction_file" && -s "$evaluation_file" ]]; then
    stage "complete test export and evaluation already exist; generation skipped"
else
    stage "starting native 10-iteration, 20-beam decoding and top-10 ranking"
    PYTHON_BIN="$python_bin" CHEM_PYTHON_BIN="$chem_python_bin" MECHET_DIR="$mechet_dir" \
        GPUS="$gpus" OUTPUT_DIR="$generation_dir" AUGMENTATION=10 \
        BEAM_SIZE=20 REPOS_BEAM=5 MASK_BEAM=1 TOKEN_BEAM=4 TOP_N=10 MAX_ITER=10 \
        "$repo_dir/scripts/run_mechet_generate.sh" \
        "$dataset" "$finetune_checkpoint" full test
fi

stage "pipeline complete: $prediction_file"
