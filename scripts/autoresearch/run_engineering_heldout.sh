#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 4 )); then
  echo "usage: $0 DATA ADAPTER OUTPUT MODEL_REVISION" >&2
  exit 2
fi

data=$1
adapter=$2
output=$3
revision=$4
python_bin=/root/miniconda3/envs/meteor/bin/python

echo "[meteor-pr69-heldout] begin data=$data adapter=$adapter output=$output revision=$revision"
"$python_bin" -u scripts/eval_natural_language_event_local.py run \
  --data "$data" --adapter "$adapter" --output "$output" \
  --model Qwen/Qwen3-0.6B --model-revision "$revision" \
  --load-mode fp16 \
  --sample-reactions 4 --seed 17 --batch-size 1 \
  --max-new-tokens 256 --max-context 4096
"$python_bin" -u scripts/eval_natural_language_event_local.py aggregate \
  --data "$data" --adapter "$adapter" --output "$output" \
  --model Qwen/Qwen3-0.6B --model-revision "$revision" \
  --load-mode fp16 \
  --sample-reactions 4 --seed 17
echo "[meteor-pr69-heldout] complete output=$output/evaluation.json"
