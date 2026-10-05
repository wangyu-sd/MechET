#!/usr/bin/env bash
set -Eeuo pipefail

repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr81-system-one-20261004
source_data=/aaa/fionafyang/buddy1/whaleywang/MechET/data/mech_uspto_31k_natural_language_history_v2
shared_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
output_root=/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/system_one_pr81_jev_typed_v2_31k_20261005
model_revision=c1899de289a04d12100db370d81485cdf75e47ca

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$repo/src:$repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wheels=$(mktemp -d /tmp/mechet_jev_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheels" \
  /aaa/fionafyang/buddy1/whaleywang/MechET/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
export PYTHONPATH="$wheels:$PYTHONPATH"

python - <<'PY'
import torch, rdkit
print({'phase':'runtime_gate','rdkit':rdkit.__version__,
       'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}, flush=True)
assert rdkit.__version__ == '2026.03.4'
assert torch.cuda.device_count() == 1
assert 'A100' in torch.cuda.get_device_name(0).upper()
PY

model_cache=$(mktemp -d /tmp/mechet_jev_hf.XXXXXX)
cp -a "$shared_cache/models--Qwen--Qwen3-0.6B" "$model_cache/"
export HF_HUB_CACHE="$model_cache"

printf '[jev-typed-v2] frozen-data/tokenization preflight\n'
python scripts/train_jev_style_electron_flow.py \
  --train "$source_data/train.jsonl" --valid "$source_data/valid.jsonl" \
  --model Qwen/Qwen3-0.6B --revision "$model_revision" \
  --output "$output_root/preflight" --audit-only

printf '[jev-typed-v2] full one-epoch training; no bounded smoke stage\n'
python scripts/train_jev_style_electron_flow.py \
  --train "$source_data/train.jsonl" --valid "$source_data/valid.jsonl" \
  --model Qwen/Qwen3-0.6B --revision "$model_revision" \
  --output "$output_root/full" --epochs 1

test -s "$output_root/full/validation_epoch1.json"
test -s "$output_root/full/typed_head_epoch1.pt"
test -s "$output_root/full/run_manifest_epoch1.json"
printf '[jev-typed-v2] training and validation complete\n'
