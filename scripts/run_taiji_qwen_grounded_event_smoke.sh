#!/usr/bin/env bash
set -Eeuo pipefail

artifact_root=${MECHET_GROUNDED_ARTIFACT_ROOT:-/aaa/fionafyang/buddy1/whaleywang/MechET}
repo_dir=${MECHET_GROUNDED_RUNTIME_DIR:-$artifact_root}
model_source=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
selected_rows=${MECHET_SMOKE_ROWS:-$artifact_root/outputs/gates/rapid_a7_rescue_20260910/selected_rows.jsonl}
run_dir=${MECHET_GROUNDED_EVENT_RUN_DIR:-$artifact_root/outputs/eval/qwen3_8b_grounded_event_smoke_20260910}
tasks=$run_dir/tasks.jsonl
predictions=$run_dir/predictions

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$selected_rows" ]]; then
  echo "missing frozen validation selection: $selected_rows" >&2
  exit 2
fi
if [[ ! -f "$model_source/config.json" ]]; then
  echo "missing pure Qwen3-8B snapshot: $model_source" >&2
  exit 2
fi

mkdir -p "$run_dir" "$predictions"
cd "$repo_dir"

python scripts/build_qwen_grounded_event_smoke.py \
  --data "$selected_rows" \
  --output "$tasks" \
  --seed 17 \
  --max-candidates 8 \
  --minimum-candidates 2 \
  --workers "${MECHET_GROUNDED_BUILD_WORKERS:-16}"

# Pure-Qwen means exactly the frozen base checkpoint: no LoRA/PEFT adapter is
# loaded anywhere in this launcher or evaluator.
torchrun --standalone --nproc_per_node=8 scripts/eval_qwen_grounded_event_smoke.py run \
  --data "$tasks" \
  --output "$predictions" \
  --model "$model_source"

python scripts/eval_qwen_grounded_event_smoke.py aggregate \
  --data "$tasks" \
  --output "$predictions" \
  --model "$model_source"
