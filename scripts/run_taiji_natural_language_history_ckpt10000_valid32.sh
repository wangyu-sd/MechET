#!/usr/bin/env bash
set -Eeuo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor

artifact_root=/aaa/fionafyang/buddy1/whaleywang/MechET
code_dir=${MECHET_HISTORY_EVAL_RUNTIME_DIR:?set MECHET_HISTORY_EVAL_RUNTIME_DIR to the detached evaluation worktree}
data=$artifact_root/data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl
state_adapter=$artifact_root/outputs/agent/natural_language_event_sft_qwen3_8b_a100_seed17_20260913
history_adapter=$artifact_root/outputs/eval/natural_language_history_ckpt10000_valid32_20260918/adapter
value_adapter=$artifact_root/outputs/agent/natural_language_state_value_v2_qwen3_8b_h20_seed17_20260915
output=$artifact_root/outputs/eval/natural_language_history_ckpt10000_valid32_20260918

cd "$code_dir"
export HF_HUB_CACHE=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$code_dir/src:$code_dir/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

check_sha256() {
  local expected=$1
  local path=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "[meteor-history-smoke] checksum mismatch path=$path expected=$expected actual=$actual" >&2
    exit 2
  fi
}

check_sha256 7303a6018850db61594af5854df936778c21a6668e61151f95e2c74c32d22d2a "$data"
check_sha256 16648e587e084c273c35faee0adcd2486fbdb4f71985d007648421ea5990f3fb "$state_adapter/adapter_model.safetensors"
check_sha256 28f3a326a64adf463b805ce05b41249c075f9995a13b59c8f5cb2a4191c3b3ef "$history_adapter/adapter_model.safetensors"
check_sha256 28ab4a29e52a11fcec227814b7b35526efdf15f8a774ae5b930e40c0039f5e82 "$value_adapter/adapter_model.safetensors"

if [[ -e "$output/state_sft" || -e "$output/trajectory_sft" || -e "$output/evaluation.json" ]]; then
  echo "[meteor-history-smoke] refusing to overwrite an existing result condition in $output" >&2
  exit 2
fi
mkdir -p "$output/state_sft" "$output/trajectory_sft"

gpu_count=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
if [[ "$gpu_count" != 8 ]]; then
  echo "[meteor-history-smoke] expected 8 visible GPUs, found $gpu_count" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "[meteor-history-smoke] development-only matched validation smoke"
echo "[meteor-history-smoke] rows=32 seed=17 test_loaded=false gold_history_visible=false"
echo "[meteor-history-smoke] GPUs 0-3=State-SFT; GPUs 4-7=compact-history Trajectory-SFT checkpoint-10000"

common_args=(
  --data "$data"
  --value-adapter "$value_adapter"
  --sample-reactions 32
  --seed 17
  --branching 1
  --early-beam 2
  --late-beam 1
  --early-depth 2
  --max-decisions 12
  --max-imports 8
  --max-new-tokens 384
  --value-weight 0.20
  --no-4bit
  --reject-target-retained-finish
  # This completed smoke used the historical v1 action-conditioned prompts.
  # Keep the flag only for exact artifact reproduction; v2 uses one prompt.
  --legacy-dual-prompt
)

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone --master_port=29621 --nproc_per_node=4 \
  scripts/run_natural_language_value_search.py \
  "${common_args[@]}" \
  --policy-adapter "$state_adapter" \
  --output "$output/state_sft" &
state_pid=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --standalone --master_port=29622 --nproc_per_node=4 \
  scripts/run_natural_language_value_search.py \
  "${common_args[@]}" \
  --policy-adapter "$history_adapter" \
  --output "$output/trajectory_sft" \
  --compact-history &
history_pid=$!

set +e
wait "$state_pid"
state_status=$?
wait "$history_pid"
history_status=$?
set -e
echo "[meteor-history-smoke] state_status=$state_status history_status=$history_status"
if (( state_status != 0 || history_status != 0 )); then
  exit 1
fi

python -u scripts/summarize_natural_language_history_smoke.py \
  --state-sft "$output/state_sft" \
  --trajectory-sft "$output/trajectory_sft" \
  --output "$output/evaluation.json"
echo "[meteor-history-smoke] complete result=$output/evaluation.json"
