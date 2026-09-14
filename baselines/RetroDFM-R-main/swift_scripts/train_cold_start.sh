#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/outputs/pretrain}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/data/cold_start/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/cold_start}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

mkdir -p "${OUTPUT_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE}" swift sft \
    --model "${MODEL_PATH}" \
    --tuner_type full \
    --dataset "${DATA_PATH}" \
    --load_from_cache_file true \
    --check_model false \
    --packing false \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --max_length 16384 \
    --save_strategy steps \
    --save_steps 1000 \
    --logging_steps 5 \
    --output_dir "${OUTPUT_DIR}" \
    --dataloader_num_workers 32 \
    --dataset_num_proc 32 \
    --save_only_model true \
    --deepspeed zero2
