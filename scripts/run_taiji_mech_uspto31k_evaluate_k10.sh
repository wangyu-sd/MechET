#!/usr/bin/env bash
set -Eeuo pipefail

echo >&2 "WARNING: INCOMPLETE TRACE-VIEW SUBSET: evaluating 1,124/3,120 mech-USPTO-31k test reactions; output is not headline-eligible."

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
reference="$repo_dir/data/mech_uspto_31k_inverse_tool_sft/test.jsonl"
predictions="$repo_dir/outputs/eval/mech_uspto31k_qwen3_8b_k10/predictions.jsonl"
report="$repo_dir/outputs/eval/mech_uspto31k_qwen3_8b_k10/evaluation.json"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
from pathlib import Path
import torch

required = [
    Path("data/mech_uspto_31k_inverse_tool_sft/test.jsonl"),
    Path("outputs/eval/mech_uspto31k_qwen3_8b_k10/predictions.jsonl"),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing frozen evaluation artifacts: {missing}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"expected exactly 8 CUDA devices, got {torch.cuda.device_count()}")
print({"cuda_devices": torch.cuda.device_count(), "torch": torch.__version__})
PY

exec python scripts/evaluate_prediction_set.py \
  --reference "$reference" \
  --predictions "$predictions" \
  --output "$report" \
  --condition-name mech_uspto31k_trace_k10 \
  --expected-rows 1124 \
  --expected-candidates 10
