#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 FINETUNE_CHECKPOINT_DIR [OUTPUT_CHECKPOINT]" >&2
    exit 2
fi

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
checkpoint_dir=$(realpath "$1")
output=${2:-$checkpoint_dir/finetune_average.pt}
average_count=${AVERAGE_COUNT:-10}

"$python_bin" "$repo_dir/utils/average_checkpoints.py" \
    --inputs "$checkpoint_dir" \
    --output "$output" \
    --num-update-checkpoints "$average_count"

echo "$output"

