#!/usr/bin/env bash
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=${MECHET_NL_V2_EVAL_RUNTIME_DIR:?set MECHET_NL_V2_EVAL_RUNTIME_DIR}
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
data=$shared_repo/data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl
adapter=$shared_repo/outputs/eval/natural_language_event_v2_ckpt14000_finegrained_20260919/adapter
output_root=$shared_repo/outputs/eval/natural_language_event_v2_ckpt14000_finegrained_20260919
local_output=$output_root/local_valid256
suffix_output=$output_root/suffix_valid32
rdkit_wheel=$shared_repo/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
bitsandbytes_wheel=$shared_repo/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared_repo"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

runtime_target=$(mktemp -d /tmp/mechet_nl_v2_eval_runtime.XXXXXX)
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $rdkit_wheel" | sha256sum --check --strict
echo "54b771f06e1a3c73af5c7f16ccf0fc23a846052813d4b008d10cb6e017dd1c8c  $bitsandbytes_wheel" | sha256sum --check --strict
python -m pip install --quiet --no-deps --target "$runtime_target" \
  "$rdkit_wheel" "$bitsandbytes_wheel"
export PYTHONPATH=$runtime_target:$runtime_repo/src:$runtime_repo

local_hf_cache=$(mktemp -d /tmp/mechet_nl_v2_eval_hf.XXXXXX)
echo "[meteor-nl-v2-eval] staging pinned Qwen3-8B cache"
cp -a "$shared_hf_cache/models--Qwen--Qwen3-8B" "$local_hf_cache/"
export HF_HUB_CACHE=$local_hf_cache

test -f "$data"
test -f "$adapter/adapter_config.json"
echo "2cd25b892b812eef0d93b99c29dd96bf73bc61098f6f54077fbb6d5439c26a09  $adapter/adapter_model.safetensors" | sha256sum --check --strict
python - <<'PY'
import rdkit, torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 8 or not all("A100" in name.upper() for name in names):
    raise SystemExit(f"expected 8 A100 GPUs, got {names}")
if rdkit.__version__ != "2026.03.4":
    raise SystemExit(f"expected RDKit 2026.03.4, got {rdkit.__version__}")
print({"gpus": names, "rdkit_version": rdkit.__version__}, flush=True)
PY

echo "[meteor-nl-v2-eval] phase=gold-state-local reactions=256 decisions=all"
torchrun --standalone --nproc_per_node=8 \
  "$runtime_repo/scripts/eval_natural_language_event_local.py" run \
  --data "$data" --output "$local_output" --model Qwen/Qwen3-8B \
  --adapter "$adapter" --sample-reactions 256 --seed 17 \
  --batch-size 2 --max-new-tokens 512 --max-context 4096 --dtype bfloat16
python -u "$runtime_repo/scripts/eval_natural_language_event_local.py" aggregate \
  --data "$data" --output "$local_output" --model Qwen/Qwen3-8B \
  --adapter "$adapter" --sample-reactions 256 --seed 17 --dtype bfloat16

echo "[meteor-nl-v2-eval] phase=closed-loop-suffix reactions=32 horizons=1,2,3,full"
torchrun --standalone --nproc_per_node=8 \
  "$runtime_repo/scripts/eval_natural_language_event_suffix.py" run \
  --data "$data" --output "$suffix_output" --model Qwen/Qwen3-8B \
  --adapter "$adapter" --sample-reactions 32 --seed 17 \
  --horizons 1 2 3 full --history-window 6 --history-mode state_only \
  --max-new-tokens 512 --max-context 4096 --dtype bfloat16
python -u "$runtime_repo/scripts/eval_natural_language_event_suffix.py" aggregate \
  --data "$data" --output "$suffix_output" --model Qwen/Qwen3-8B \
  --adapter "$adapter" --sample-reactions 32 --seed 17 \
  --horizons 1 2 3 full --history-window 6 --history-mode state_only \
  --dtype bfloat16

echo "[meteor-nl-v2-eval] complete local=$local_output/evaluation.json suffix=$suffix_output/suffix_evaluation.json"
