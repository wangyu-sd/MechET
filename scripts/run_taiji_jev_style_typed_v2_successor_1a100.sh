#!/usr/bin/env bash
set -Eeuo pipefail

repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr81-system-one-20261004
shared=/aaa/fionafyang/buddy1/whaleywang/MechET
source_data="$shared/data/mech_uspto_31k_natural_language_history_v2"
checkpoint="$shared/outputs/agent/system_one_pr81_jev_typed_v2_31k_20261005/full"
output_root="$shared/outputs/agent/system_one_pr81_jev_typed_v2_successor_20261005"
shared_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$repo/src:$repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wheels=$(mktemp -d /tmp/mechet_jev_eval_wheels.XXXXXX)
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

test -s "$checkpoint/run_manifest_epoch1.json"
test -s "$checkpoint/typed_head_epoch1.pt"
test -s "$checkpoint/adapter_epoch1/adapter_model.safetensors"
test ! -e "$output_root/valid"
test ! -e "$output_root/test"

model_cache=$(mktemp -d /tmp/mechet_jev_eval_hf.XXXXXX)
printf '[jev-typed-v2-eval] staging pinned model\n'
cp -a "$shared_cache/models--Qwen--Qwen3-0.6B" "$model_cache/"
export HF_HUB_CACHE="$model_cache"

for split in valid test; do
  printf '[jev-typed-v2-eval] starting %s full local successor replay\n' "$split"
  python scripts/eval_jev_style_successor.py \
    --checkpoint "$checkpoint" \
    --data "$source_data/$split.jsonl" \
    --split "$split" \
    --output "$output_root/$split" \
    --log-every 100
  test -s "$output_root/$split/report.json"
  test -s "$output_root/$split/cases.jsonl"
  printf '[jev-typed-v2-eval] %s replay report verified\n' "$split"
done
