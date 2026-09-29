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

printf '[meteor-r3-localization] code_commit=%s data_root=%s\n' "$(git rev-parse HEAD)" "$data_root"
exec /root/miniconda3/envs/meteor/bin/python -u \
  scripts/autoresearch/run_r3_localization_inference.py \
  --prompts "$data_root/outputs/autoresearch/prepared_eval/r3_localization_prompts_v1_20260929/r3_localization_prompts.jsonl" \
  --queries "$data_root/outputs/autoresearch/prepared_eval/r3_flower_unmarked_localization_v1_20260929/r3_unmarked_queries.jsonl" \
  --adapter "$data_root/outputs/agent/natural_language_event_history_v2_qwen3_8b_h20_seed17_20260918" \
  --base-model-path "$base_model" \
  --output "$data_root/outputs/autoresearch/prepared_eval/r3_stageii_localization_predictions_v1_20260929" \
  --seed 17 --max-new-tokens 64 --max-context 8192
