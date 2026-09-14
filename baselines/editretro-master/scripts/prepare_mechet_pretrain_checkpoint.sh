#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 PRETRAIN_CHECKPOINT_DIR [OUTPUT_CHECKPOINT]" >&2
    exit 2
fi

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
checkpoint_dir=$(realpath "$1")
output=${2:-$checkpoint_dir/pretrain_for_finetune.pt}
average_count=${AVERAGE_COUNT:-5}
raw_average="$checkpoint_dir/pretrain_average_raw.pt"

"$python_bin" "$repo_dir/utils/average_checkpoints.py" \
    --inputs "$checkpoint_dir" \
    --output "$raw_average" \
    --num-update-checkpoints "$average_count"

"$python_bin" "$repo_dir/utils/pretrain_ckpt_utils.py" \
    --inputckpt "$raw_average" \
    --outputckpt "$output"

echo "$output"
