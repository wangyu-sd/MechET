#!/usr/bin/env bash
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=${MECHET_NL_EVENT_RUNTIME_DIR:?set MECHET_NL_EVENT_RUNTIME_DIR}
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
data=$shared_repo/data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl
adapter=$shared_repo/outputs/agent/natural_language_event_sft_qwen3_8b_a100_seed17_20260913
output=$shared_repo/outputs/eval/natural_language_event_suffix_state_only_valid32_final_k1_a100_20260914
rdkit_wheel=$shared_repo/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
bitsandbytes_wheel=$shared_repo/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared_repo"
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$shared_hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

runtime_target=$(mktemp -d /tmp/mechet_nl_suffix_state.XXXXXX)
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $rdkit_wheel" | sha256sum --check --strict
echo "54b771f06e1a3c73af5c7f16ccf0fc23a846052813d4b008d10cb6e017dd1c8c  $bitsandbytes_wheel" | sha256sum --check --strict
python -m pip install --quiet --no-deps --target "$runtime_target" "$rdkit_wheel" "$bitsandbytes_wheel"
export PYTHONPATH=$runtime_target:$runtime_repo/src:$runtime_repo

test -f "$data"
test -f "$adapter/adapter_config.json"
echo "16648e587e084c273c35faee0adcd2486fbdb4f71985d007648421ea5990f3fb  $adapter/adapter_model.safetensors" | sha256sum --check --strict
python - <<'PY'
import rdkit, torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 8 or not all("A100" in name.upper() for name in names):
    raise SystemExit(f"expected 8 A100 GPUs, got {names}")
if rdkit.__version__ != "2026.03.4":
    raise SystemExit(f"expected RDKit 2026.03.4, got {rdkit.__version__}")
print({"gpus": names, "rdkit_version": rdkit.__version__}, flush=True)
PY

echo "[natural-language-state-only] reactions=32 horizons=1,2,3,full mode=state_only"
torchrun --standalone --nproc_per_node=8 \
  "$runtime_repo/scripts/eval_natural_language_event_suffix.py" run \
  --data "$data" --output "$output" --model Qwen/Qwen3-8B \
  --adapter "$adapter" --sample-reactions 32 --seed 17 \
  --horizons 1 2 3 full --history-mode state_only \
  --max-new-tokens 512 --max-context 4096
python -u "$runtime_repo/scripts/eval_natural_language_event_suffix.py" aggregate \
  --data "$data" --output "$output" --model Qwen/Qwen3-8B \
  --adapter "$adapter" --sample-reactions 32 --seed 17 \
  --horizons 1 2 3 full --history-mode state_only
echo "[natural-language-state-only] complete report=$output/suffix_evaluation.json"
