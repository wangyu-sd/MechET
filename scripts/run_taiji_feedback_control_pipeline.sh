#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=${MECHET_REPO_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-controls}
artifact_root=${MECHET_ARTIFACT_ROOT:-/aaa/fionafyang/buddy1/whaleywang/MechET}
control=${MECHET_CONTROL_PIPELINE:?set MECHET_CONTROL_PIPELINE}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"
export PYTHONUNBUFFERED=1

case "$control" in
  b1_no_enumeration) config=configs/iclr/b1_no_enumeration_sft.yaml; trace_control=b1_no_enumeration ;;
  b3_stale_feedback) config=configs/iclr/b3_stale_feedback_sft.yaml; trace_control=b3_stale_feedback ;;
  b5_direct_legal_actions) config=configs/iclr/b5_direct_legal_actions_sft.yaml ;;
  *) echo >&2 "unsupported MECHET_CONTROL_PIPELINE=$control"; exit 2 ;;
esac

python scripts/build_flower_feedback_controls.py \
  --control "$control" \
  --source-dir "$artifact_root/data/flower_inverse_tool_sft_compact_full_state_v1"

echo "[meteor-progress] stage=${control}-training-start time=$(date --iso-8601=seconds)"
MECHET_REPO_DIR="$repo_dir" MECHET_ARTIFACT_ROOT="$artifact_root" \
MECHET_TRAINING_CONFIG="$config" MECHET_EXPECTED_GPU=A100 \
bash scripts/run_taiji_iclr_full_sft.sh
echo "[meteor-progress] stage=${control}-training-complete time=$(date --iso-8601=seconds)"

if [[ "$control" == b5_direct_legal_actions ]]; then
  export MECHET_BASELINE=direct_legal_actions
  export MECHET_INFERENCE_RUNNER="$repo_dir/scripts/run_taiji_iclr_full_inference.sh"
else
  export MECHET_TRACE_CONTROL="$trace_control"
  export MECHET_INFERENCE_RUNNER="$repo_dir/scripts/run_taiji_trace_control_inference.sh"
fi
export MECHET_REPO_DIR="$repo_dir"
export MECHET_ARTIFACT_ROOT="$artifact_root"
export MECHET_EXPECTED_GPU=A100
export SAMPLES_PER_TARGET=${SAMPLES_PER_TARGET:-1}
exec bash scripts/bootstrap_taiji_vllm_a100.sh
