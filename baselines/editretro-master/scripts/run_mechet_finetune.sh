#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 {mech_uspto_31k_full|flower_full} PRETRAIN_CHECKPOINT [full|audit100]" >&2
    exit 2
fi

dataset=$1
pretrain_checkpoint=$(realpath "$2")
profile=${3:-full}
case "$dataset" in
    mech_uspto_31k_full|flower_full) ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac
case "$profile" in
    full|audit100) ;;
    *) echo "Unsupported profile: $profile" >&2; exit 2 ;;
esac

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
cpu=${CPU:-0}
if command -v fairseq-train >/dev/null 2>&1; then
    fairseq_train=(fairseq-train)
else
    export PYTHONPATH="$repo_dir/fairseq:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
    fairseq_train=("$python_bin" "$repo_dir/fairseq/fairseq_cli/train.py")
fi
native_dataset=$dataset
if [[ "$profile" == "audit100" ]]; then
    native_dataset="${dataset}_audit100"
fi
artifact_dir="$repo_dir/datasets/$native_dataset/aug${AUGMENTATION:-10}"
"$python_bin" "$repo_dir/utils/require_validated_artifact.py" "$artifact_dir"
if [[ ! -s "$pretrain_checkpoint" ]]; then
    echo "Missing pretrain checkpoint: $pretrain_checkpoint" >&2
    exit 1
fi

restore_checkpoint=$pretrain_checkpoint
restore_args=(--reset-optimizer --reset-lr-scheduler --reset-meters --reset-dataloader)
tee_args=()
resume_checkpoint=${RESUME_CHECKPOINT:-}
resume_bool=false
if [[ -n "$resume_checkpoint" ]]; then
    restore_checkpoint=$(realpath "$resume_checkpoint")
    if [[ ! -s "$restore_checkpoint" ]]; then
        echo "Missing resume checkpoint: $restore_checkpoint" >&2
        exit 1
    fi
    restore_args=()
    tee_args=(-a)
    resume_bool=true
fi

run_name=${RUN_NAME:-$(date -u "+%Y%m%d_%H%M%S")}
checkpoint_dir=${CHECKPOINT_DIR:-$repo_dir/results/mechet/$dataset/$profile/finetune/$run_name/checkpoints}
mkdir -p "$checkpoint_dir"
override_dir=${RUNTIME_OVERRIDE_DIR:-$repo_dir/runtime_overrides/$dataset/$run_name}

gpus=${GPUS:-0}
override_active=0
if [[ -s "$override_dir/finetune_gpus.txt" ]]; then
    IFS= read -r gpus < "$override_dir/finetune_gpus.txt"
    gpus=${gpus//[[:space:]]/}
    override_active=1
fi
if [[ ! "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Invalid finetune GPU list: $gpus" >&2
    exit 2
fi
gpu_list=${gpus//,/ }
read -r -a gpu_ids <<< "$gpu_list"
gpu_count=${#gpu_ids[@]}
runtime_args=()
precision_args=(--fp16)
launcher=(env CUDA_VISIBLE_DEVICES="$gpus" CUDA_LAUNCH_BLOCKING=1)
if [[ "$cpu" == "1" ]]; then
    gpu_count=1
    runtime_args+=(--cpu)
    precision_args=()
    launcher=(env)
fi

lr=${LR:-0.0003}
max_tokens=${MAX_TOKENS:-40960}
if [[ -s "$override_dir/finetune_max_tokens.txt" ]]; then
    IFS= read -r max_tokens < "$override_dir/finetune_max_tokens.txt"
    max_tokens=${max_tokens//[[:space:]]/}
    override_active=1
fi
if [[ ! "$max_tokens" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid finetune max-tokens value: $max_tokens" >&2
    exit 2
fi
warmup_updates=${WARMUP_UPDATES:-10000}
update_freq=${UPDATE_FREQ:-1}
max_epoch=${MAX_EPOCH:-2000}
max_update=${MAX_UPDATE:-2000000}
save_interval_updates=${SAVE_INTERVAL_UPDATES:-10000}
keep_interval_updates=${KEEP_INTERVAL_UPDATES:--1}
log_interval=${LOG_INTERVAL:-500}
num_workers=${NUM_WORKERS:-1}
no_epoch_checkpoints=${NO_EPOCH_CHECKPOINTS:-0}
no_save=${NO_SAVE:-0}
save_args=()
if [[ "$no_epoch_checkpoints" == "1" ]]; then
    save_args+=(--no-epoch-checkpoints)
fi
if [[ "$no_save" == "1" ]]; then
    save_args+=(--no-save)
fi

if [[ "$override_active" == "1" ]]; then
    echo "Using finetune runtime override from $override_dir: gpus=$gpus max_tokens=$max_tokens"
fi

cat > "$checkpoint_dir/run_config.json" <<EOF
{"stage":"finetune","dataset":"$dataset","profile":"$profile","data_bin":"$artifact_dir/data-bin","pretrain_checkpoint":"$pretrain_checkpoint","restore_checkpoint":"$restore_checkpoint","resume":$resume_bool,"gpus":"$gpus","cpu":$cpu,"lr":$lr,"max_tokens":$max_tokens,"warmup_updates":$warmup_updates,"update_freq":$update_freq,"max_epoch":$max_epoch,"max_update":$max_update,"save_interval_updates":$save_interval_updates,"keep_interval_updates":$keep_interval_updates,"log_interval":$log_interval,"num_workers":$num_workers,"no_epoch_checkpoints":$no_epoch_checkpoints,"no_save":$no_save,"alpha_ratio":0.5,"dae_ratio":0.5,"seed":1}
EOF

"${launcher[@]}" "${fairseq_train[@]}" \
    "$artifact_dir/data-bin" \
    --user-dir "$repo_dir/editretro" \
    -s src -t tgt \
    --save-dir "$checkpoint_dir" \
    --ddp-backend no_c10d \
    --task translation_retro \
    --criterion nat_loss \
    --arch editretro_nat \
    --noise random_delete_shuffle \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr "$lr" --lr-scheduler inverse_sqrt \
    --min-lr 1e-09 \
    --warmup-updates "$warmup_updates" \
    --warmup-init-lr 1e-07 \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --share-all-embeddings \
    --decoder-learned-pos --encoder-learned-pos \
    --max-tokens-valid 4000 \
    --log-format simple --log-interval "$log_interval" \
    --num-workers "$num_workers" \
    --fixed-validation-seed 7 \
    --max-tokens "$max_tokens" \
    --save-interval-updates "$save_interval_updates" \
    --keep-interval-updates "$keep_interval_updates" \
    --keep-last-epochs 20 \
    "${save_args[@]}" \
    --max-epoch "$max_epoch" \
    --max-update "$max_update" \
    --alpha-ratio 0.5 \
    --dae-ratio 0.5 \
    "${precision_args[@]}" \
    --update-freq "$update_freq" \
    --restore-file "$restore_checkpoint" \
    "${restore_args[@]}" \
    --seed 1 \
    --distributed-world-size "$gpu_count" \
    "${runtime_args[@]}" \
    2>&1 | tee "${tee_args[@]}" "$checkpoint_dir/finetune.log"

echo "$checkpoint_dir"
