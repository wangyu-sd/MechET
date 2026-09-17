#!/usr/bin/env bash
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=${MECHET_HISTORY_SFT_RUNTIME_DIR:?set MECHET_HISTORY_SFT_RUNTIME_DIR}
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
source_dir=$shared_repo/data/flower_natural_language_event_sft_v1_rdkit2026
output_dir=$shared_repo/data/flower_natural_language_event_history_v1
training_config=$runtime_repo/configs/agent/natural_language_event_history_qwen3_8b_h20.yaml
parent_adapter=$shared_repo/outputs/agent/natural_language_event_sft_qwen3_8b_a100_seed17_20260913
liger_wheel=$shared_repo/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl
xformers_wheel=$shared_repo/artifacts/wheels/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl
bitsandbytes_wheel=$shared_repo/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared_repo"

echo "[history-sft] runtime=$runtime_repo source=$source_dir output=$output_dir"
test -f "$training_config"
test -f "$output_dir/manifest.json"
echo "16648e587e084c273c35faee0adcd2486fbdb4f71985d007648421ea5990f3fb  $parent_adapter/adapter_model.safetensors" | sha256sum --check --strict
echo "303b9bbf5c10f9289c3139afb41e4d989e8c809516624a106b89b064163d971d  $liger_wheel" | sha256sum --check --strict
echo "bbf2f500dfdbcf4649bf568cc2c9f434399f704dc4064fd1fbdbef2b524a8139  $xformers_wheel" | sha256sum --check --strict
echo "54b771f06e1a3c73af5c7f16ccf0fc23a846052813d4b008d10cb6e017dd1c8c  $bitsandbytes_wheel" | sha256sum --check --strict

runtime_target=$(mktemp -d /tmp/mechet_history_runtime.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" \
  "$liger_wheel" "$xformers_wheel" "$bitsandbytes_wheel"

export HF_HUB_CACHE=$shared_hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$runtime_target:$runtime_repo/src:$runtime_repo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

python - <<'PY'
import rdkit
import torch
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(i) for i in range(8)]
if not all("H20" in name.upper() for name in names):
    raise SystemExit(f"expected H20 GPUs, got {names}")
print({"gpus": names, "rdkit_version": rdkit.__version__}, flush=True)
PY

python - "$output_dir/manifest.json" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())
assert m['status'] == 'validated_complete' and m['training_allowed'] is True
assert m['reaction_denominator'] == {'train': 257167, 'valid': 2890, 'test': 28967}
assert m['decision_rows'] == {'train': 2007421, 'valid': 22341, 'test': 225613}
assert m['decision_contract'] == 'compressed_history_tool_decision_v1'
assert m['history_contract'] == 'executor_compact_accepted_actions_v1'
assert m['gold_standard_only'] is True and int(m['failed_actions']) == 0
assert m['model_visible_gold_horizon'] is False
print({'dataset_gate': 'passed', 'reactions': m['reaction_denominator'], 'decisions': m['decision_rows']}, flush=True)
PY

cache_manifest=$output_dir/qwen3_8b_tokens_4096/manifest.json
if [[ ! -f "$cache_manifest" ]]; then
  echo "[history-sft] distributed tokenization starts"
  torchrun --standalone --nproc_per_node=8 \
    "$runtime_repo/scripts/prepare_tool_sft_arrow.py" --config "$training_config"
fi

python - "$cache_manifest" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())
assert int(m['splits']['train']['n_rows']) == 2007421
assert int(m['splits']['validation']['n_rows']) == 22341
assert int(m['splits']['train'].get('truncation_count', 0)) == 0
print({'token_gate': 'passed', 'train': m['splits']['train']}, flush=True)
PY

local_cache=$(mktemp -d /tmp/mechet_history_tokens_4096.XXXXXX)
echo "[history-sft] staging token cache to node-local storage: $local_cache"
cp -a "$output_dir/qwen3_8b_tokens_4096/." "$local_cache/"
export MECHET_PRETOKENIZED_CACHE_DIR=$local_cache
echo "[history-sft] local token cache ready bytes=$(du -sb "$local_cache" | cut -f1)"

local_hf_cache=$(mktemp -d /tmp/mechet_hf_cache.XXXXXX)
echo "[history-sft] staging pinned Qwen3-8B cache to node-local storage: $local_hf_cache"
cp -a "$shared_hf_cache/models--Qwen--Qwen3-8B" "$local_hf_cache/"
export HF_HUB_CACHE=$local_hf_cache
echo "[history-sft] local model cache ready bytes=$(du -sb "$local_hf_cache" | cut -f1)"

echo "[history-sft] one-epoch continuation from verified natural-language-event SFT starts"
exec torchrun --standalone --nproc_per_node=8 \
  "$runtime_repo/scripts/train_tool_sft.py" --config "$training_config"
