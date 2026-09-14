#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
SLIME_DIR=/root/slime

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export OPENAI_API_KEY=None
export OPENAI_BASE_URL=http://grm-host:8000/v1
export MASTER_ADDR=127.0.0.1

NUM_GPUS=4
MODEL_DIR="${PROJECT_DIR}/outputs/cold_start"
HF_SAVE_DIR="${PROJECT_DIR}/outputs/rl/iter_{rollout_id}"
PROMPT_DATA="${PROJECT_DIR}/data/rl/full/train.jsonl"
EVAL_DATA="${PROJECT_DIR}/data/eval/full.jsonl"
NUM_ROLLOUT=3000
WANDB_GROUP=full-grm-qwen3-8b

source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"

if nvidia-smi topo -m 2>/dev/null | grep -qE 'NV[0-9]'; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}"
    --load "${MODEL_DIR}"
    --save-interval 200
    --save-hf "${HF_SAVE_DIR}"
    --save-hf-only
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key input
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --custom-rm-path scripts.retrodfm.reward.custom_grm
    --reward-key rewards
    --custom-rollout-log-function-path scripts.retrodfm.reward.log_rollout_data
    --over-sampling-batch-size 128
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size 64
    --n-samples-per-prompt 8
    --rollout-max-response-len 4096
    --rollout-temperature 1
    --global-batch-size 512
    --balance-data
)

EVAL_ARGS=(
    --eval-interval 200
    --eval-prompt-data retror "${EVAL_DATA}"
    --custom-eval-rollout-log-function-path scripts.retrodfm.reward.log_eval_rollout_data
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 4096
    --eval-top-p 1
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-tis
    --use-kl-loss
    --kl-loss-coef 0.001
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project retrodfm
    --wandb-group "${WANDB_GROUP}"
)

SGLANG_ARGS=(
    --rollout-num-gpus 2
    --rollout-num-gpus-per-engine 1
    --sglang-server-concurrency 128
    --sglang-log-level warning
    --sglang-mem-fraction-static 0.8
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

cd "${SLIME_DIR}"
ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host 0.0.0.0 \
    --dashboard-port 8265
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

RUNTIME_ENV_JSON="$(printf '{\"env_vars\":{\"PYTHONPATH\":\"/root/Megatron-LM:%s:%s\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"%s\",\"WANDB_MODE\":\"offline\",\"OPENAI_API_KEY\":\"None\",\"OPENAI_BASE_URL\":\"%s\",\"GRM_MODEL\":\"Qwen3.6-35B-A3B\",\"GRM_CONCURRENCY\":\"512\"}}' \
    "${SLIME_DIR}" \
    "${PROJECT_DIR}" \
    "${HAS_NVLINK}" \
    "${OPENAI_BASE_URL}")"

ray job submit --address http://127.0.0.1:8265 \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    -- python3 train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 2 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"
