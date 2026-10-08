#!/usr/bin/env bash
set -Eeuo pipefail

shared=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime=/aaa/fionafyang/buddy1/whaleywang/MechET-pr82-reliable-20261006
hf=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
data=$shared/data/mech_uspto_31k_natural_language_event_v2
parent=$shared/outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17
config=$runtime/configs/agent/reliable_mech_uspto31k_state_06b_v100.yaml
output=$shared/outputs/agent/reliable_mech_uspto31k_state_qwen3_0_6b_v100_seed17

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH=$runtime/src:$runtime
export HF_HUB_CACHE=$hf

echo "[31k-06b] FlowER State-SFT -> 31k State-SFT, validation early stopping"
echo "[31k-06b] runtime=$(git -C "$runtime" rev-parse HEAD) config=$config"
python - "$data" "$parent" "$config" "$output" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch, yaml

data, parent, config, output = map(Path, sys.argv[1:])
cfg = yaml.safe_load(config.read_text())
manifest = json.loads((data / 'manifest.json').read_text())
cache = json.loads((data / 'qwen3_0_6b_tokens_4096/manifest.json').read_text())
expected = {'train': 10152, 'valid': 1319, 'test': 1253}
assert manifest['training_allowed'] and manifest['status'] == 'validated_trace_view'
assert manifest['reaction_denominator'] == expected == cfg['contract']['reaction_denominator']
assert manifest['full_reaction_denominator'] == {'train':24959,'valid':3120,'test':3120}
assert manifest['decision_rows'] == {'train':32401,'valid':4288,'test':4006}
assert manifest['observation_contract'] == cfg['contract']['observation_contract']
assert manifest['decision_contract'] == cfg['contract']['decision_contract']
assert cache['model_name_or_path'] == cfg['model_name_or_path'] == 'Qwen/Qwen3-0.6B'
assert cache['model_revision'] == cfg['training']['model_revision']
assert cache['derived_from_identical_tokenizer_cache']['reuse_policy'] == 'byte_identical_tokenizer_assets_and_frozen_source_bytes_v1'
for split, key in [('train','train'),('valid','validation')]:
    path = data / f'{split}.jsonl'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == manifest['splits'][split]['output_sha256'] == cache['sources'][key]['sha256']
    assert cache['splits'][key]['n_rows'] == manifest['decision_rows'][split]
    assert cache['splits'][key]['truncation_count'] == 0
assert Path(cfg['initial_adapter_path']) == parent
assert Path(cfg['output_dir']) == output
assert (parent / 'adapter_model.safetensors').is_file()
digest = hashlib.sha256((parent / 'adapter_model.safetensors').read_bytes()).hexdigest()
assert digest == 'fbd8db06094fd8029d4cb0cac38cbb88280a442d76f686258194723019b1a1bd'
assert cfg['training']['fp16'] and not cfg['training']['bf16']
assert cfg['training']['eval_strategy'] == 'epoch' and cfg['training']['early_stopping_patience'] == 2
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names) == 8 and all('V100' in name.upper() for name in names), names
print({'gate':'passed','parent_sha256':digest,'reactions':expected,'decisions':manifest['decision_rows'],'gpus':names},flush=True)
PY

if [[ -s "$output/adapter_model.safetensors" ]]; then
  echo "[31k-06b] final output exists; refusing duplicate run" >&2
  exit 3
fi

local_tokens=$(mktemp -d /tmp/mechet_31k_06b_tokens.XXXXXX)
cp -aL "$data/qwen3_0_6b_tokens_4096/." "$local_tokens/"
export MECHET_PRETOKENIZED_CACHE_DIR=$local_tokens
local_hf=$(mktemp -d /tmp/mechet_31k_06b_hf.XXXXXX)
cp -a "$hf/models--Qwen--Qwen3-0.6B" "$local_hf/"
export HF_HUB_CACHE=$local_hf

echo "[31k-06b] optimizer starts: up to 8 epochs, stop after 2 non-improving validation epochs"
torchrun --standalone --nproc_per_node=8 "$runtime/scripts/train_tool_sft.py" --config "$config" \
  ${MECHET_RELIABLE_RESUME:+--resume-from-checkpoint}
test -s "$output/adapter_model.safetensors"
test -s "$output/adapter_manifest.json"
sha256sum "$output/adapter_model.safetensors"
echo "[31k-06b] completed"
