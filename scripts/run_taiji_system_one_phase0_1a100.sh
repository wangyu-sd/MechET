#!/usr/bin/env bash
set -Eeuo pipefail

repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr81-system-one-20261004
source_data=/aaa/fionafyang/buddy1/whaleywang/MechET/data/mech_uspto_31k_natural_language_history_v2
shared_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
output_root=/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/system_one_pr81_phase0_31k_20261005_v2
model_revision=c1899de289a04d12100db370d81485cdf75e47ca

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$repo/src:$repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wheels=$(mktemp -d /tmp/mechet_system_one_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheels" \
  /aaa/fionafyang/buddy1/whaleywang/MechET/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
export PYTHONPATH="$wheels:$PYTHONPATH"
python - <<'PY'
import rdkit
print({'phase': 'rdkit_gate', 'rdkit': rdkit.__version__}, flush=True)
assert rdkit.__version__ == '2026.03.4'
PY

printf '[system-one] launcher=pr81_system_one_phase0_v2\n'
python - <<'PY'
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print({'phase': 'gpu_gate', 'gpu_count': len(names), 'gpu_names': names}, flush=True)
assert len(names) == 1 and 'A100' in names[0].upper()
PY

model_cache=$(mktemp -d /tmp/mechet_system_one_hf.XXXXXX)
printf '[system-one] staging pinned 0.6B model\n'
cp -a "$shared_cache/models--Qwen--Qwen3-0.6B" "$model_cache/"
export HF_HUB_CACHE="$model_cache"

printf '[system-one] full split preflight\n'
python scripts/train_system_one_electron_flow.py \
  --train "$source_data/train.jsonl" --valid "$source_data/valid.jsonl" \
  --model Qwen/Qwen3-0.6B --revision "$model_revision" \
  --output "$output_root/preflight" --audit-only

printf '[system-one] bounded 16-event optimizer and validation smoke\n'
python scripts/train_system_one_electron_flow.py \
  --train "$source_data/train.jsonl" --valid "$source_data/valid.jsonl" \
  --model Qwen/Qwen3-0.6B --revision "$model_revision" \
  --output "$output_root/smoke16" --max-train-events 16 --valid-limit 16 \
  --epochs 1
test -s "$output_root/smoke16/validation_epoch1.json"
test -s "$output_root/smoke16/decision_head_epoch1.pt"

printf '[system-one] full 19,199-event Phase-0 optimizer starting\n'
python scripts/train_system_one_electron_flow.py \
  --train "$source_data/train.jsonl" --valid "$source_data/valid.jsonl" \
  --model Qwen/Qwen3-0.6B --revision "$model_revision" \
  --output "$output_root/full" --epochs 1
test -s "$output_root/full/validation_epoch1.json"
test -s "$output_root/full/decision_head_epoch1.pt"
printf '[system-one] full Phase-0 validation complete\n'
