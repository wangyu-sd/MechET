#!/usr/bin/env bash
set -Eeuo pipefail

worktree=/aaa/fionafyang/buddy1/whaleywang/MechET-pr69-implementation
data_root=/aaa/fionafyang/buddy1/whaleywang/MechET
base_model=/aaa/fionafyang/buddy1/whaleywang/models/Qwen3-8B

cd "$worktree"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

printf '[meteor-r3-repair] code_commit=%s data_root=%s\n' "$(git rev-parse HEAD)" "$data_root"
exec /root/miniconda3/envs/meteor/bin/python -u \
  scripts/autoresearch/run_r3_repair_inference.py \
  --prompts "$data_root/outputs/autoresearch/prepared_eval/r3_repair_prompts_v1_20260929/r3_repair_prompts.jsonl" \
  --queries "$data_root/outputs/autoresearch/prepared_eval/r3_flower_queries_v1_20260929/r3_queries.jsonl" \
  --adapter "$data_root/outputs/agent/natural_language_event_history_v2_qwen3_8b_h20_seed17_20260918" \
  --base-model-path "$base_model" \
  --output "$data_root/outputs/autoresearch/prepared_eval/r3_stageii_repair_predictions_v1_20260929" \
  --load-mode fp16 --seed 17 --max-new-tokens 512 --max-context 8192
