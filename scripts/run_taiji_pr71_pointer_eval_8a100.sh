#!/usr/bin/env bash
set -Eeuo pipefail

repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr71-pointer-20261001
source_data=/aaa/fionafyang/buddy1/whaleywang/MechET/data/mech_uspto_31k_natural_language_history_v2
adapter=/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/mech_uspto31k_nl_history_v2_qwen3_8b_h20_seed17_20260922
checkpoint=/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/pr71_p0_pointer_31k_stage2_8a100_20261001/pointer_head_epoch1.pt
output=/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/pr71_pointer_pair_valid_8a100_20261001
model_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$repo/src:$repo"

python - <<'PY'
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print({'phase': 'gpu_gate', 'gpu_count': len(names), 'gpu_names': names}, flush=True)
assert len(names) == 8 and all('A100' in name.upper() for name in names)
PY

wheels=$(mktemp -d /tmp/mechet_pr71_eval_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheels" \
  /aaa/fionafyang/buddy1/whaleywang/MechET/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
export PYTHONPATH="$wheels:$PYTHONPATH"
python scripts/train_electron_pointer.py \
  --data-dir "$source_data" --adapter "$adapter" --output "$output" --audit-only

staged_cache=$(mktemp -d /tmp/mechet_pr71_eval_hf.XXXXXX)
echo '[pr71-pointer-eval] staging pinned Qwen3-8B model locally' >&2
cp -a "$model_cache/models--Qwen--Qwen3-8B" "$staged_cache/"
export HF_HUB_CACHE="$staged_cache"

echo '[pr71-pointer-eval] 8-rank full valid paired-move evaluation starting' >&2
torchrun --standalone --nproc_per_node=8 scripts/train_electron_pointer.py \
  --data-dir "$source_data" --adapter "$adapter" \
  --eval-checkpoint "$checkpoint" --output "$output" \
  --epochs 1 --max-length 4096
test -s "$output/validation_epoch1.json"
echo '[pr71-pointer-eval] paired and multi-flow validation complete' >&2
