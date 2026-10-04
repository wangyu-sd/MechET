#!/usr/bin/env bash
set -Eeuo pipefail

repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr81-system-one-20261004
shared=/aaa/fionafyang/buddy1/whaleywang/MechET
data_dir="$shared/data/mech_uspto_31k_natural_language_history_v2"
adapter="$shared/outputs/agent/mech_uspto31k_nl_history_v2_qwen3_8b_h20_seed17_20260922"
checkpoint="$shared/outputs/agent/pr71_conditional_pointer_31k_stage2_8a100_20261001"
output="$shared/outputs/agent/pr81_pr71_matched_successor_20261005"
shared_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$repo/src:$repo"

wheels=$(mktemp -d /tmp/mechet_pr81_pr71_eval_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheels" \
  "$shared/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
export PYTHONPATH="$wheels:$PYTHONPATH"
python - <<'PY'
import rdkit
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print({'phase': 'runtime_gate', 'rdkit': rdkit.__version__,
       'gpu_count': len(names), 'gpu_names': names}, flush=True)
assert rdkit.__version__ == '2026.03.4'
assert len(names) == 1 and 'A100' in names[0].upper()
PY

python scripts/eval_pr71_pointer_successor.py \
  --checkpoint "$checkpoint" --adapter "$adapter" --data-dir "$data_dir" \
  --output "$output" --split all --audit-only

staged_cache=$(mktemp -d /tmp/mechet_pr81_pr71_eval_hf.XXXXXX)
printf '[pr81-pr71-comparator] staging pinned 8B model\n'
cp -a "$shared_cache/models--Qwen--Qwen3-8B" "$staged_cache/"
export HF_HUB_CACHE="$staged_cache"

printf '[pr81-pr71-comparator] frozen valid and test local successor evaluation starting\n'
python scripts/eval_pr71_pointer_successor.py \
  --checkpoint "$checkpoint" --adapter "$adapter" --data-dir "$data_dir" \
  --output "$output" --split all --log-every 100
test -s "$output/valid/report.json"
test -s "$output/valid/cases.jsonl"
test -s "$output/test/report.json"
test -s "$output/test/cases.jsonl"
printf '[pr81-pr71-comparator] full local successor evaluation complete\n'
