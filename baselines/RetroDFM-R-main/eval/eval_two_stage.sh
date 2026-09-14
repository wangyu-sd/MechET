#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${1:?Usage: $0 MODEL_PATH [OUTPUT_NAME] [DATA_PATH] [TASK] [N_THINK] [N_ANSWER] [AUGMENTATION] [BEAM_SIZE]}"
OUTPUT_NAME="${2:-$(basename "${MODEL_PATH}")}"
DATA_PATH="${3:-${PROJECT_DIR}/data/eval/two_stage.jsonl}"
TASK="${4:-uspto_50k_R}"
N_THINK="${5:-10}"
N_ANSWER="${6:-10}"
AUGMENTATION="${7:-20}"
BEAM_SIZE="${8:-$((N_THINK * N_ANSWER))}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
HOST="${SERVER_HOST:-0.0.0.0}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
PORT="${API_PORT:-30000}"
DP_SIZE="${DP_SIZE:-2}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/eval/${OUTPUT_NAME}"
mkdir -p "${OUTPUT_DIR}"

python -m sglang_router.launch_server \
    --model-path "${MODEL_PATH}" \
    --dp-size "${DP_SIZE}" \
    --router-policy round_robin \
    --host "${HOST}" \
    --port "${PORT}" \
    --log-level error \
    --log-level-http error \
    --router-log-level error &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

BASE_URL="http://${CLIENT_HOST}:${PORT}/v1"
until curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; do sleep 5; done

python "${SCRIPT_DIR}/generate_two_stage.py" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --task "${TASK}" \
    --base_url "${BASE_URL}" \
    --n_think "${N_THINK}" \
    --n_answer "${N_ANSWER}" \
    --think_temperature "${THINK_TEMPERATURE:-1.1}" \
    --answer_temperature "${ANSWER_TEMPERATURE:-1.4}" \
    --think_max_tokens "${THINK_MAX_TOKENS:-2048}" \
    --answer_max_tokens "${ANSWER_MAX_TOKENS:-256}" \
    --concurrency "${CONCURRENCY:-2048}" \
    --answer_concurrency "${ANSWER_CONCURRENCY:-256}"

kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
trap - EXIT

python "${SCRIPT_DIR}/score_rsmiles.py" \
    --output_dir "${OUTPUT_DIR}" \
    --task "${TASK}" \
    --augmentation "${AUGMENTATION}" \
    --beam_size "${BEAM_SIZE}" \
    --alpha 0.0
