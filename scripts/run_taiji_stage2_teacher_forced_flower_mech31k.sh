#!/usr/bin/env bash
# Frozen Stage-II Qwen3-0.6B teacher-forced FlowER + mech-USPTO-31k smoke.
# Only frozen inference: no SFT, optimizer, RSI, or paper edits.
set -Eeuo pipefail

shared=${MECHET_SHARED_REPO:-/aaa/fionafyang/buddy1/whaleywang/MechET}
code=${MECHET_STAGE2_TF_CODE_DIR:-$PWD}
adapter=${MECHET_STAGE2_TF_ADAPTER:-$shared/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17}
adapter_sha=40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75
runner="$code/scripts/eval_mechet_stage2_teacher_forced.py"
output=${MECHET_STAGE2_TF_OUTPUT:-$shared/outputs/eval/stage2_teacher_forced_flower_mech31k_20261022}
flower="$shared/data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl"
mech31k="$shared/data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/valid.jsonl"

for required in "$runner" "$flower" "$mech31k" \
  "$adapter/adapter_model.safetensors" "$adapter/adapter_config.json"; do
  if [[ ! -f "$required" ]]; then
    echo "[stage2-tf] MISSING: $required" >&2
    exit 2
  fi
done

export PYTHONPATH="$code/src:$code${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# The 0.6B model can be evaluated on a single available GPU.
# This is intentionally not an 8xA100 training job.
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("stage2-tf requires one available CUDA GPU")
print("[stage2-tf] cuda_device", torch.cuda.get_device_name(0), flush=True)
PY

flower_n=${MECHET_FLOWER_SMOKE_N:-64}
mech_n=${MECHET_MECH31K_SMOKE_N:-32}
mkdir -p "$output"
echo "[stage2-tf] corpus=FlowER n=$flower_n"
echo "[stage2-tf] corpus=mech-USPTO-31k n=$mech_n"
echo "[stage2-tf] read_only_stage2_adapter=$adapter"

run_one() {
  local label="$1" corpus="$2" data="$3" expected="$4" n="$5"
  local prepared="$output/$label/prepared"
  local predictions="$output/$label/predictions"
  if [[ -e "$prepared" ]]; then
    echo "[stage2-tf] refusing to overwrite previous preparation: $prepared" >&2
    exit 2
  fi
  mkdir -p "$output/$label"

  echo "[stage2-tf] $label: replay/authenticated-state preflight"
  python -u "$runner" prepare \
    --corpus "$corpus" --data "$data" --expected-source-rows "$expected" \
    --sample-reactions "$n" --seed 17 --output "$prepared"

  echo "[stage2-tf] $label: greedy next-action inference on frozen 0.6B"
  torchrun --standalone --nproc_per_node=1 "$runner" run \
    --input "$prepared" --output "$predictions" \
    --model Qwen/Qwen3-0.6B \
    --revision c1899de289a04d12100db370d81485cdf75e47ca \
    --adapter "$adapter" --expected-adapter-sha "$adapter_sha" \
    --dtype bfloat16 --batch-size 4 \
    --max-new-tokens 512 --max-context 16384

  echo "[stage2-tf] $label: full-denominator scored report"
  python -u "$runner" aggregate --input "$prepared" --output "$predictions"
}

run_one "flower" "flower_new_dataset" "$flower" 2890 "$flower_n"
run_one "mech31k" "mech_uspto_31k" "$mech31k" 1319 "$mech_n"
echo "[stage2-tf] finished: $output"
