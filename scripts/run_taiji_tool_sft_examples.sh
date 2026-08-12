#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
adapter="$repo_dir/outputs/agent/tool_sft_mixed_inverse_qwen3_8b"
output="$repo_dir/outputs/examples/mixed_inverse_qwen3_8b_heldout4.jsonl"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"

export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

test -f "$adapter/adapter_manifest.json"
test "$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)" -eq 1

exec python scripts/infer_mechet.py \
  --config configs/agent/tool_sft_mixed_inverse_qwen3_8b.yaml \
  --data data/mixed_inverse_tool_sft/test.jsonl \
  --output "$output" \
  --mode trace \
  --condition-name mixed_inverse_heldout_greedy_smoke \
  --adapter "$adapter" \
  --samples-per-target 1 \
  --max-iterations 12 \
  --max-new-tokens 512 \
  --temperature 0 \
  --id mech-uspto31k-inverse:12443 \
  --id mech-uspto31k-inverse:1409 \
  --id textbook-tool-sft:flower_mech_proof_test_28065 \
  --id textbook-tool-sft:flower_mech_proof_test_11142
