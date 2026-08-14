#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
predictions="$repo_dir/outputs/eval/mech_uspto31k_qwen3_8b_k10/predictions.jsonl"
reference="$repo_dir/data/mech_uspto_31k_inverse_tool_sft/test.jsonl"
adapter="$repo_dir/outputs/agent/tool_sft_mixed_inverse_qwen3_8b"
output_dir="$repo_dir/outputs/eval/mech_uspto31k_qwen3_8b_k10/nll_ranking_v1"
revision=b968826d9c46dd6066d109eabc6255188de91218

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"
export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
from pathlib import Path
import torch
required = [
    Path("data/mech_uspto_31k_inverse_tool_sft/test.jsonl"),
    Path("outputs/eval/mech_uspto31k_qwen3_8b_k10/predictions.jsonl"),
    Path("outputs/agent/tool_sft_mixed_inverse_qwen3_8b/adapter_manifest.json"),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing frozen ranking artifacts: {missing}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"expected exactly 8 CUDA devices, got {torch.cuda.device_count()}")
print({"cuda_devices": torch.cuda.device_count(), "torch": torch.__version__})
PY

mkdir -p "$output_dir"
pids=()
for gpu in $(seq 0 7); do
  shard=$(printf '%02d' "$gpu")
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/score_and_rank_predictions.py score \
    --predictions "$predictions" \
    --output "$output_dir/nll_scores.shard-${shard}.jsonl" \
    --model Qwen/Qwen3-8B \
    --revision "$revision" \
    --adapter "$adapter" \
    --shard-count 8 \
    --shard-index "$gpu" \
    --max-length 24576 \
    >"$output_dir/shard-${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  tail -n 100 "$output_dir"/shard-*.log
  exit "$status"
fi

exec python scripts/score_and_rank_predictions.py aggregate \
  --reference "$reference" \
  --predictions "$predictions" \
  --ranking-dir "$output_dir" \
  --output "$output_dir/evaluation.json"
