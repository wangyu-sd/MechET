#!/usr/bin/env bash
set -Eeuo pipefail

# Run an isolated kernel benchmark from a completed A7 checkpoint. This script
# never writes into the source training directory and never stops the live job.
repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
source_output=${MECHET_BENCHMARK_SOURCE_OUTPUT:-outputs/agent/tool_sft_flower_full_qwen3_8b_a100_run_20260819}
benchmark_steps=${MECHET_BENCHMARK_STEPS:-100}
benchmark_root=${MECHET_BENCHMARK_ROOT:-outputs/benchmarks/a7_flash_sdpa_liger_full}
training_config=configs/agent/tool_sft_flower_full_qwen3_8b_h20_fast.yaml

cd "$repo_dir"
# The Taiji base image does not put the training interpreter on PATH.  This
# wrapper needs Python before it delegates to the common runner (to inspect the
# source trainer_state.json), so activate the same environment here as well.
conda_sh=/root/miniconda3/etc/profile.d/conda.sh
if [[ ! -r "$conda_sh" ]]; then
  echo "missing Conda activation script: $conda_sh" >&2
  exit 2
fi
source "$conda_sh"
conda activate meteor
command -v python >/dev/null 2>&1 || {
  echo "python is unavailable after activating Conda environment 'meteor'" >&2
  exit 2
}
[[ "$benchmark_steps" =~ ^[1-9][0-9]*$ ]] || {
  echo "MECHET_BENCHMARK_STEPS must be a positive integer" >&2
  exit 2
}

checkpoint=${MECHET_BENCHMARK_CHECKPOINT:-}
if [[ -z "$checkpoint" ]]; then
  checkpoint=$(find "$source_output" -maxdepth 1 -type d -name 'checkpoint-*' -print \
    | sort -V | tail -n 1)
fi
if [[ -z "$checkpoint" || ! -f "$checkpoint/trainer_state.json" ]]; then
  echo "no complete source checkpoint found under $source_output" >&2
  exit 2
fi
for required in adapter_config.json adapter_model.safetensors optimizer.pt scheduler.pt; do
  test -f "$checkpoint/$required" || {
    echo "incomplete source checkpoint: missing $checkpoint/$required" >&2
    exit 2
  }
done

source_step=$(python - "$checkpoint/trainer_state.json" <<'PY'
import json
from pathlib import Path
import sys
print(int(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["global_step"]))
PY
)
target_step=$((source_step + benchmark_steps))
benchmark_output="$benchmark_root/from_${source_step}_to_${target_step}"
if [[ -e "$benchmark_output" ]]; then
  echo "benchmark output already exists: $benchmark_output" >&2
  exit 2
fi

export MECHET_EXPECTED_GPU=${MECHET_EXPECTED_GPU:-H20}
export MECHET_TRAINING_CONFIG="$training_config"
export MECHET_RESUME_FROM_CHECKPOINT="$checkpoint"
export MECHET_MAX_STEPS="$target_step"
export MECHET_OUTPUT_DIR="$benchmark_output"
export MECHET_STAGE_TO_LOCAL=${MECHET_STAGE_TO_LOCAL:-1}

echo "source_checkpoint=$checkpoint"
echo "benchmark_steps=$benchmark_steps target_step=$target_step"
echo "benchmark_output=$benchmark_output"
if [[ "${MECHET_BENCHMARK_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
exec bash scripts/run_taiji_flower_full_qwen3_8b_v100.sh
