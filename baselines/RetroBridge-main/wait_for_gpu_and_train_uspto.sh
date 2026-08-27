#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_DIR/.venv-retrobridge/bin/python"
CONFIG="$PROJECT_DIR/configs/mechet_retrobridge_mech_uspto_31k_full.yaml"
DATA_ROOT="/data/pxy/data/RetroBridge/mech_uspto_31k_full"
AUDIT_REPORT="$DATA_ROOT/full_preprocess_report.json"
MPL_CONFIG_DIR="${MPLCONFIGDIR:-/tmp/retrobridge-mpl}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-60}"
LOCK_FILE="${LOCK_FILE:-/tmp/retrobridge-mech-uspto-train.lock}"
MONITORED_GPU_IDS=(0 1 2 3 4 5 6 7)

if [[ ! "$POLL_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_INTERVAL_SECONDS must be a positive integer" >&2
  exit 2
fi

for required_path in "$PYTHON" "$CONFIG" "$AUDIT_REPORT"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    echo "Run the USPTO conversion and graph audit before this script." >&2
    exit 1
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not available" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1 || ! jq -e '.passed == true' "$AUDIT_REPORT" >/dev/null; then
  echo "RetroBridge graph audit has not passed: $AUDIT_REPORT" >&2
  exit 1
fi

# Let an outer flock process retain the lock without passing its descriptor to
# PyTorch DataLoader workers. The lock is released when the main process exits.
if [[ "${RETROBRIDGE_LOCK_HELD:-0}" != "1" ]]; then
  set +e
  flock --exclusive --nonblock --conflict-exit-code 75 --close \
    "$LOCK_FILE" env RETROBRIDGE_LOCK_HELD=1 "$0" "$@"
  status=$?
  set -e
  if [[ "$status" -eq 75 ]]; then
    echo "Another USPTO RetroBridge waiter or training process holds $LOCK_FILE" >&2
  fi
  exit "$status"
fi

is_monitored_gpu() {
  local candidate="$1" monitored_gpu

  for monitored_gpu in "${MONITORED_GPU_IDS[@]}"; do
    if [[ "$candidate" == "$monitored_gpu" ]]; then
      return 0
    fi
  done

  return 1
}

find_available_gpu() {
  local stats gpu_index memory_used memory_total

  if ! stats="$(nvidia-smi \
    --query-gpu=index,memory.used,memory.total \
    --format=csv,noheader,nounits)"; then
    return 2
  fi

  while IFS=',' read -r gpu_index memory_used memory_total; do
    gpu_index="${gpu_index//[[:space:]]/}"
    memory_used="${memory_used//[[:space:]]/}"
    memory_total="${memory_total//[[:space:]]/}"

    if ! [[ "$gpu_index" =~ ^[0-9]+$ \
      && "$memory_used" =~ ^[0-9]+$ \
      && "$memory_total" =~ ^[1-9][0-9]*$ ]]; then
      continue
    fi
    if ! is_monitored_gpu "$gpu_index"; then
      continue
    fi
    if [[ $((memory_used * 100)) -lt "$memory_total" ]]; then
      printf '%s\n' "$gpu_index"
      return 0
    fi
  done <<< "$stats"

  return 1
}

echo "Monitoring GPUs ${MONITORED_GPU_IDS[*]} for memory usage below 1% (poll every ${POLL_INTERVAL_SECONDS}s)."

while true; do
  if gpu_index="$(find_available_gpu)"; then
    echo "[$(date '+%F %T')] GPU $gpu_index is below 1% memory usage; starting training."
    mkdir -p "$MPL_CONFIG_DIR"
    cd "$PROJECT_DIR"
    exec env \
      CUDA_VISIBLE_DEVICES="$gpu_index" \
      MPLCONFIGDIR="$MPL_CONFIG_DIR" \
      "$PYTHON" -u train.py \
        --config "$CONFIG" \
        --model RetroBridge \
        --disable_swanlab
  else
    status=$?
    if [[ "$status" -eq 2 ]]; then
      echo "[$(date '+%F %T')] nvidia-smi query failed; retrying." >&2
    else
      echo "[$(date '+%F %T')] No GPU is below 1% memory usage; waiting."
    fi
  fi

  sleep "$POLL_INTERVAL_SECONDS"
done
