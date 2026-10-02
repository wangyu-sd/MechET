#!/usr/bin/env bash
# Train exactly one frozen Issue #79 H2 condition on one eight-GPU node.
set -Eeuo pipefail

repo_dir=${MECHET_REPO_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET}
training_config=${MECHET_NMI_CONFIG:?set MECHET_NMI_CONFIG to a frozen H2 config}
expected_gpu=${MECHET_EXPECTED_GPU:-A100}
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
wheels_dir=${MECHET_WHEELS_DIR:-$repo_dir/artifacts/wheels}
liger_wheel="$wheels_dir/liger_kernel-0.6.2-py3-none-any.whl"
bitsandbytes_wheel="$wheels_dir/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo_dir"
export HF_HUB_CACHE="$shared_hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "303b9bbf5c10f9289c3139afb41e4d989e8c809516624a106b89b064163d971d  $liger_wheel" | sha256sum --check --strict
echo "54b771f06e1a3c73af5c7f16ccf0fc23a846052813d4b008d10cb6e017dd1c8c  $bitsandbytes_wheel" | sha256sum --check --strict
runtime_target=$(mktemp -d /tmp/mechet_nmi_sft_runtime.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" "$liger_wheel" "$bitsandbytes_wheel"
export PYTHONPATH="$runtime_target:$PYTHONPATH"

python - "$training_config" "$expected_gpu" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text())
condition = config['condition_name'].split('_qwen3_8b_seed17')[0].removeprefix('nmi_h2_')
if condition not in {'direct', 'open_flow', 'closed_loop'}:
    raise SystemExit(f'not an H2 matched config: {condition}')
config_manifest = json.loads((config_path.parent / 'manifest.json').read_text())
actual_config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
if actual_config_sha != config_manifest['config_sha256'][condition]:
    raise SystemExit('frozen H2 config SHA mismatch')
verification_path = config_path.parent.parent / 'verification.json'
if hashlib.sha256(verification_path.read_bytes()).hexdigest() != config_manifest['representation_verification_sha256']:
    raise SystemExit('H2 mapped-product/endpoint parity verification SHA mismatch')
data_manifest = json.loads(Path(config['contract']['stable_id_manifest']).read_text())
if data_manifest['condition'] != condition or not data_manifest['training_allowed']:
    raise SystemExit('H2 condition/data manifest mismatch')
if data_manifest['parent_split_manifest_sha256'] != config_manifest['split_manifest_sha256']:
    raise SystemExit('H2 split manifest mismatch')
for split, config_key in (('train', 'train_file'), ('valid', 'validation_file'), ('test', 'test_file')):
    path = Path(config[config_key])
    if not path.is_file():
        raise SystemExit(f'H2 {split} data missing: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != data_manifest['output_sha256'][split]:
        raise SystemExit(f'H2 {split} data SHA mismatch')
    if int(data_manifest['rows'][split]) != int(config['contract'][{
        'train': 'expected_train_rows',
        'valid': 'expected_validation_rows',
        'test': 'expected_test_rows',
    }[split]]):
        raise SystemExit(f'H2 {split} row contract mismatch')
if int(config['training']['max_steps']) != int(config_manifest['equal_optimizer_updates']):
    raise SystemExit('H2 optimizer-update budget mismatch')
if torch.cuda.device_count() != 8:
    raise SystemExit(f'expected 8 GPUs, got {torch.cuda.device_count()}')
names = [torch.cuda.get_device_name(i) for i in range(8)]
if not all(sys.argv[2].upper() in name.upper() for name in names):
    raise SystemExit(f'expected {sys.argv[2]} GPUs, got {names}')
print({'condition': condition, 'split_sha256': config_manifest['split_manifest_sha256'],
       'rows': data_manifest['rows'], 'steps': config['training']['max_steps'], 'gpus': names}, flush=True)
PY

run_with_heartbeat() {
  local stage=$1
  shift
  "$@" &
  local child=$!
  while kill -0 "$child" 2>/dev/null; do
    echo "[meteor-progress] stage=nmi-${stage} config=${training_config} pid=${child} time=$(date --iso-8601=seconds)"
    sleep 60
  done
  wait "$child"
}

token_cache_manifest=$(python - "$training_config" <<'PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg['pretokenized_cache_dir']) / 'manifest.json')
PY
)
if [[ ! -f "$token_cache_manifest" ]]; then
  run_with_heartbeat pretokenization torchrun --standalone --nproc_per_node=8 \
    scripts/prepare_tool_sft_arrow.py --config "$training_config"
fi

resume_args=()
run_output=$(python - "$training_config" <<'PY'
from pathlib import Path
import sys
import yaml
print(yaml.safe_load(Path(sys.argv[1]).read_text())['output_dir'])
PY
)
if find "$run_output" -maxdepth 2 -name trainer_state.json -print -quit 2>/dev/null | grep -q .; then
  resume_args+=(--resume-from-checkpoint)
fi
run_with_heartbeat training torchrun --standalone --nproc_per_node=8 \
  scripts/train_tool_sft.py --config "$training_config" "${resume_args[@]}"
