#!/usr/bin/env bash
set -Eeuo pipefail

repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr81-system-one-20261004
source_data=/aaa/fionafyang/buddy1/whaleywang/MechET/data/mech_uspto_31k_natural_language_history_v2
shared_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
checkpoint=/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/system_one_pr81_phase0_31k_20261005_v2/full
output=/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/system_one_pr81_phase0_successor_valid_20261005

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$repo/src:$repo"

test -s "$checkpoint/validation_epoch1.json"
test -s "$checkpoint/run_manifest_epoch1.json"
test -s "$checkpoint/decision_head_epoch1.pt"
test -s "$checkpoint/adapter_epoch1/adapter_model.safetensors"

wheels=$(mktemp -d /tmp/mechet_system_one_eval_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheels" \
  /aaa/fionafyang/buddy1/whaleywang/MechET/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
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

model_cache=$(mktemp -d /tmp/mechet_system_one_eval_hf.XXXXXX)
printf '[system-one-eval] staging pinned 0.6B model\n'
cp -a "$shared_cache/models--Qwen--Qwen3-0.6B" "$model_cache/"
export HF_HUB_CACHE="$model_cache"

printf '[system-one-eval] full frozen-valid successor evaluation starting\n'
python scripts/eval_system_one_successor.py \
  --checkpoint "$checkpoint" \
  --valid "$source_data/valid.jsonl" \
  --output "$output" \
  --log-every 100
test -s "$output/report.json"
test -s "$output/cases.jsonl"
printf '[system-one-eval] full frozen-valid successor evaluation complete\n'
