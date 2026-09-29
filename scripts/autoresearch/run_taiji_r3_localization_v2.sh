#!/usr/bin/env bash
set -Eeuo pipefail

worktree=/aaa/fionafyang/buddy1/whaleywang/MechET-pr69-implementation
data_root=/aaa/fionafyang/buddy1/whaleywang/MechET
base_model=/aaa/fionafyang/buddy1/whaleywang/models/Qwen3-8B
expected_script_sha=5025bc8b7586188b9ebb0527befb103c55d939512bd564e5c7760c33c8c21a40

cd "$worktree"
actual_script_sha="$(sha256sum scripts/autoresearch/run_r3_localization_inference.py | cut -d' ' -f1)"
if [[ "$actual_script_sha" != "$expected_script_sha" ]]; then
  printf '[meteor-r3-localization] inference script hash drifted: %s\n' "$actual_script_sha" >&2
  exit 1
fi
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

printf '[meteor-r3-localization] code_commit=%s script_sha256=%s data_root=%s\n' \
  "$(git rev-parse HEAD)" "$actual_script_sha" "$data_root"
exec /root/miniconda3/envs/meteor/bin/python -u \
  scripts/autoresearch/run_r3_localization_inference.py \
  --prompts "$data_root/outputs/autoresearch/prepared_eval/r3_localization_prompts_v1_20260929/r3_localization_prompts.jsonl" \
  --queries "$data_root/outputs/autoresearch/prepared_eval/r3_flower_unmarked_localization_v1_20260929/r3_unmarked_queries.jsonl" \
  --adapter "$data_root/outputs/agent/natural_language_event_history_v2_qwen3_8b_h20_seed17_20260918" \
  --base-model-path "$base_model" \
  --output "$data_root/outputs/autoresearch/prepared_eval/r3_stageii_localization_predictions_v2_20260929" \
  --seed 17 --max-new-tokens 64 --max-context 8192
