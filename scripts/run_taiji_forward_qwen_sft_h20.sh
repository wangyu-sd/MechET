#!/usr/bin/env bash
# Remote complete data conversion, tokenizer audit, then an isolated forward LoRA.
set -Eeuo pipefail
shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
source_repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr83-greedy
: "${MECHET_FORWARD_RUNTIME_SHA:?require immutable runtime commit}"
git -C "$source_repo" cat-file -e "$MECHET_FORWARD_RUNTIME_SHA^{commit}"
test -d /aaa/fionafyang/buddy1/whaleywang/datasets/retro/data/flower_new_dataset
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
runtime_repo=$(mktemp -d /tmp/mechet_forward_code.XXXXXX)
git -C "$source_repo" archive "$MECHET_FORWARD_RUNTIME_SHA" | tar -x -C "$runtime_repo"
runtime_deps=$(mktemp -d /tmp/mechet_forward_deps.XXXXXX)
rdkit_wheel=$shared_repo/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $rdkit_wheel" | sha256sum --check --strict
python -m pip install --no-deps --no-index --target "$runtime_deps" "$rdkit_wheel"
export PYTHONPATH=$runtime_deps:$runtime_repo/src:$runtime_repo
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
shared_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
local_hf=$(mktemp -d /tmp/mechet_forward_hf.XXXXXX)
cp -aL "$shared_cache/models--Qwen--Qwen3-0.6B" "$local_hf/"
export HF_HUB_CACHE=$local_hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$runtime_repo"
config_template=$runtime_repo/configs/agent/forward_event_flowER_qwen3_0_6b_h20_20261013.yaml
data_dir=$shared_repo/data/flower_forward_qwen_event_v1_20261013
rendered_config=$data_dir/training_config.yaml
export MECHET_FORWARD_CONFIG_TEMPLATE=$config_template
python - <<'PY'
import os, rdkit, torch, yaml
from pathlib import Path
from mechet.forward_qwen_events import check_adapter_isolation
cfg = yaml.safe_load(Path(os.environ['MECHET_FORWARD_CONFIG_TEMPLATE']).read_text())
contract = cfg['contract']
check_adapter_isolation(Path(contract['protected_reverse_adapter']), Path(cfg['output_dir']), contract['protected_reverse_weights_sha256'])
assert not cfg.get('initial_adapter_path')
assert not Path(cfg['output_dir'], 'adapter_model.safetensors').exists(), 'forward adapter already exists'
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names) == 8 and all('H20' in n.upper() for n in names), names
assert torch.cuda.is_bf16_supported()
assert rdkit.__version__ == '2026.03.4', rdkit.__version__
print({'phase':'forward-preflight-passed', 'gpus':names, 'protected_reverse_sha':contract['protected_reverse_weights_sha256']}, flush=True)
PY
echo '[forward-sft] preparing all strict FlowER reactions with server CPUs'
python -u scripts/build_forward_qwen_event_sft.py \
  --raw-dir /aaa/fionafyang/buddy1/whaleywang/datasets/retro/data/flower_new_dataset \
  --proof-dir "$shared_repo/data/mechet_proof_sft_flower_full_v4" \
  --output-dir "$data_dir" --config-template "$config_template" \
  --rendered-config "$rendered_config" --workers "${MECHET_FORWARD_CPU_WORKERS:-64}"
echo '[forward-sft] distributed full-data tokenization with explicit assistant masks'
torchrun --standalone --nproc_per_node=8 scripts/prepare_tool_sft_arrow.py --config "$rendered_config"
python - "$rendered_config" <<'PY'
import json, sys, yaml
from pathlib import Path
from mechet.forward_qwen_events import validate_forward_token_cache
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
manifest = json.loads(Path(cfg['pretokenized_cache_dir'], 'manifest.json').read_text())
dataset = json.loads(Path(cfg['contract']['stable_id_manifest']).read_text())
for split, name in [('train','train'),('validation','valid')]:
    audit=manifest['splits'][split]
    validate_forward_token_cache(audit, dataset['splits'][name]['rows'])
    assert manifest['sources'][split]['sha256'] == dataset['splits'][name]['output_sha256']
print({'phase':'full-forward-token-audit-passed', 'train_rows':manifest['splits']['train']['n_rows'],
       'validation_rows':manifest['splits']['validation']['n_rows']}, flush=True)
PY
cache_bytes=$(du -sb "$data_dir/qwen3_0_6b_tokens_16384" | cut -f1)
free_bytes=$(df -PB1 /tmp | awk 'NR==2 {print $4}')
if (( free_bytes > cache_bytes + 4294967296 )); then
  local_cache=$(mktemp -d /tmp/mechet_forward_tokens.XXXXXX)
  cp -aL "$data_dir/qwen3_0_6b_tokens_16384/." "$local_cache/"
  export MECHET_PRETOKENIZED_CACHE_DIR=$local_cache
  echo "[forward-sft] cache staged node-locally bytes=$cache_bytes"
else
  echo "[forward-sft] node disk free=$free_bytes cache=$cache_bytes; using sealed shared Arrow mmap without duplicating cache"
fi
echo '[forward-sft] starting TWO full epochs; global batch 64; frozen reverse branch untouched'
torchrun --standalone --nproc_per_node=8 scripts/train_tool_sft.py --config "$rendered_config"
python - "$rendered_config" <<'PY'
import json, sys, yaml
from pathlib import Path
from mechet.forward_qwen_events import check_adapter_isolation
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
contract = cfg['contract']
check_adapter_isolation(Path(contract['protected_reverse_adapter']), Path(cfg['output_dir']), contract['protected_reverse_weights_sha256'])
assert Path(cfg['output_dir'], 'adapter_model.safetensors').is_file()
Path(cfg['output_dir'], 'forward_protocol_contract.json').write_text(json.dumps(contract, indent=2))
print({'phase':'forward-sft-completed', 'output_dir':cfg['output_dir'], 'reverse_weights_unchanged':True}, flush=True)
PY
