#!/usr/bin/env bash
set -Eeuo pipefail

repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr81-system-one-20261004
shared=/aaa/fionafyang/buddy1/whaleywang/MechET
data_dir="$shared/data/mech_uspto_31k_natural_language_history_v2"
checkpoint="$shared/outputs/agent/system_one_pr81_phase0_31k_20261005_v2/full"
output="$shared/outputs/agent/system_one_pr81_phase1a_route_31k_20261005"
shared_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$repo/src:$repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wheels=$(mktemp -d /tmp/mechet_system_one_route_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheels" \
  "$shared/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
export PYTHONPATH="$wheels:$PYTHONPATH"
python - <<'PY'
import rdkit
import torch
print({'phase': 'runtime_gate', 'rdkit': rdkit.__version__,
       'gpu_count': torch.cuda.device_count(),
       'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
      flush=True)
assert rdkit.__version__ == '2026.03.4'
assert torch.cuda.device_count() == 1
assert 'A100' in torch.cuda.get_device_name(0).upper()
PY

model_cache=$(mktemp -d /tmp/mechet_system_one_route_hf.XXXXXX)
printf '[system-one] staging pinned Qwen3-0.6B model\n'
cp -a "$shared_cache/models--Qwen--Qwen3-0.6B" "$model_cache/"
export HF_HUB_CACHE="$model_cache"

printf '[system-one] Phase-1a all-split action-family training starting\n'
python scripts/train_system_one_action_family.py \
  --data-dir "$data_dir" \
  --checkpoint "$checkpoint" \
  --output "$output" \
  --batch-size 8 --max-length 4096 --epochs 30 --seed 17
test -s "$output/report.json"
test -s "$output/action_family_head.pt"
printf '[system-one] Phase-1a report and head verified\n'
