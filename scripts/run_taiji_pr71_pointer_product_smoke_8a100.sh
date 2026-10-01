#!/usr/bin/env bash
set -Eeuo pipefail

repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr71-pointer-20261001
source_data=/aaa/fionafyang/buddy1/whaleywang/MechET/data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/valid.jsonl
adapter=/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/mech_uspto31k_nl_history_v2_qwen3_8b_h20_seed17_20260922
pointer=/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/pr71_conditional_pointer_31k_stage2_8a100_20261001/pointer_head_epoch1.pt
output_base=/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/eval/pr71_pointer_product_start_valid16_greedyfix_20261001
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

wheels=$(mktemp -d /tmp/mechet_pr71_product_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheels" \
  /aaa/fionafyang/buddy1/whaleywang/MechET/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
export PYTHONPATH="$wheels:$PYTHONPATH"

python - <<PY
import hashlib
import json
from pathlib import Path
source = Path('$source_data')
manifest = json.loads((source.parent / 'manifest.json').read_text())
print({'phase': 'data_gate', 'source': str(source),
       'valid_reactions': manifest.get('splits', {}).get('valid', {}).get('rows')}, flush=True)
assert source.is_file() and Path('$pointer').is_file()
assert manifest['splits']['valid']['rows'] == 1319
assert manifest['validation_passed'] and manifest['tokenizer_audit_passed']
assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['splits']['valid']['sha256']
PY

staged_cache=$(mktemp -d /tmp/mechet_pr71_product_hf.XXXXXX)
echo '[pr71-product-smoke] staging pinned Qwen3-8B model locally' >&2
cp -a "$model_cache/models--Qwen--Qwen3-8B" "$staged_cache/"
export HF_HUB_CACHE="$staged_cache"

common=(--data "$source_data" --policy-adapter "$adapter" --model Qwen/Qwen3-8B
  --sample-reactions 16 --seed 17 --reaction-level --compact-history
  --vnext-v2-prefix --no-4bit --search-no-value --value-weight 0
  --early-beam 1 --late-beam 1 --max-decisions 12
  --max-imports 12 --max-new-tokens 384)

echo '[pr71-product-smoke] pure product-only greedy K=1' >&2
torchrun --standalone --nproc_per_node=8 scripts/run_natural_language_value_search.py \
  "${common[@]}" --branching 1 --output "$output_base/greedy_k1"

echo '[pr71-product-smoke] product-only K=4 NLL baseline' >&2
torchrun --standalone --nproc_per_node=8 scripts/run_natural_language_value_search.py \
  "${common[@]}" --branching 4 --output "$output_base/baseline_k4"

echo '[pr71-product-smoke] coupled pointer reranking, same products and search budget' >&2
torchrun --standalone --nproc_per_node=8 scripts/run_natural_language_value_search.py \
  "${common[@]}" --branching 4 --pointer-head "$pointer" --pointer-weight 0.2 \
  --output "$output_base/coupled_pointer_k4"

echo '[pr71-product-smoke] greedy and both matched K4 product-only arms completed' >&2
