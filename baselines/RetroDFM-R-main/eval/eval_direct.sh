#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${1:?Usage: $0 MODEL_PATH [OUTPUT_NAME] [DATA_PATH] [TASK] [N] [AUGMENTATION] [BEAM_SIZE] [TEMPERATURE] [MAX_TOKENS]}"
OUTPUT_NAME="${2:-$(basename "${MODEL_PATH}")}"
DATA_PATH="${3:-${PROJECT_DIR}/data/eval/direct.jsonl}"
TASK="${4:-uspto_50k_R}"
N="${5:-1}"
AUGMENTATION="${6:-1}"
BEAM_SIZE="${7:-${N}}"
TEMPERATURE="${8:-0.6}"
MAX_TOKENS="${9:-4096}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HOST="${SERVER_HOST:-0.0.0.0}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
PORT="${API_PORT:-30000}"
DP_SIZE="${DP_SIZE:-1}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/eval/${OUTPUT_NAME}"
mkdir -p "${OUTPUT_DIR}"

python -m sglang_router.launch_server \
    --model-path "${MODEL_PATH}" \
    --dp-size "${DP_SIZE}" \
    --router-policy cache_aware \
    --host "${HOST}" \
    --port "${PORT}" \
    --log-level error \
    --log-level-http error \
    --router-log-level error &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

BASE_URL="http://${CLIENT_HOST}:${PORT}/v1"
until curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; do sleep 5; done

python "${SCRIPT_DIR}/generate.py" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --task "${TASK}" \
    --base_url "${BASE_URL}" \
    --n "${N}" \
    --temperature "${TEMPERATURE}" \
    --max_tokens "${MAX_TOKENS}" \
    --concurrency "${CONCURRENCY:-512}"

kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
trap - EXIT

python "${SCRIPT_DIR}/score_rsmiles.py" \
    --output_dir "${OUTPUT_DIR}" \
    --task "${TASK}" \
    --augmentation "${AUGMENTATION}" \
    --beam_size "${BEAM_SIZE}" \
    --alpha 0.0
