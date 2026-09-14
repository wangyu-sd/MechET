#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.6-35B-A3B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-35B-A3B}"
PORT="${PORT:-8000}"

python -m sglang_router.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --dp-size 4 \
    --reasoning-parser qwen3 \
    --mamba-scheduler-strategy extra_buffer \
    --mem-fraction-static 0.8 \
    --router-policy cache_aware \
    --host 0.0.0.0 \
    --port "${PORT}"

