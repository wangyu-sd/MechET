#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/data/pretrain/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/pretrain}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

mkdir -p "${OUTPUT_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE}" megatron sft \
    --model "${MODEL_PATH}" \
    --save_safetensors true \
    --dataset "${DATA_PATH}" \
    --load_from_cache_file true \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --micro_batch_size 8 \
    --global_batch_size 64 \
    --torch_dtype bfloat16 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 3e-5 \
    --lr_warmup_fraction 0.05 \
    --lr_decay_style constant \
    --num_train_epochs 3 \
    --output_dir "${OUTPUT_DIR}" \
    --save_steps 1000 \
    --max_length 16384 \
    --dataloader_num_workers 64 \
    --dataset_num_proc 64 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash

