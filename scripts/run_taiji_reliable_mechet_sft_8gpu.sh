#!/usr/bin/env bash
# Run exactly one frozen Qwen3-0.6B SFT stage on H20 or A100.
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr82-reliable-20261006
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
stage=${MECHET_RELIABLE_STAGE:?set MECHET_RELIABLE_STAGE=state or trajectory}
expected_gpu=${MECHET_RELIABLE_EXPECTED_GPU:-H20}
case "$expected_gpu" in H20|A100) ;; *) echo "unsupported GPU kind=$expected_gpu" >&2; exit 2 ;; esac

case "$stage" in
  state)
    config=$runtime_repo/configs/agent/natural_language_event_v2_qwen3_0_6b.yaml
    data_dir=$shared_repo/data/flower_natural_language_event_sft_v2
    output=$shared_repo/outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17
    ;;
  trajectory)
    config=$runtime_repo/configs/agent/natural_language_history_v2_qwen3_0_6b.yaml
    data_dir=$shared_repo/data/flower_natural_language_event_history_v2
    output=$shared_repo/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17
    : "${MECHET_RELIABLE_EXPECTED_PARENT_SHA256:?freeze the Stage-I adapter SHA first}"
    ;;
  *) echo "invalid MECHET_RELIABLE_STAGE=$stage" >&2; exit 2 ;;
esac

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared_repo"
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$shared_hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$runtime_repo/src:$runtime_repo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[reliable-mechet] stage=$stage runtime=$(git -C "$runtime_repo" rev-parse HEAD)"
echo "[reliable-mechet] data=$data_dir output=$output"
test -f "$config"
test -f "$data_dir/manifest.json"
test -f "$data_dir/ARTIFACT_STATUS.json"
test -f "$data_dir/qwen3_0_6b_tokens_4096/manifest.json"
test -d "$shared_hf_cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
if [[ -s $output/adapter_model.safetensors ]]; then
  echo "[reliable-mechet] final adapter already exists; refusing duplicate training" >&2
  exit 3
fi

if [[ $stage == trajectory ]]; then
  parent=$shared_repo/outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17
  python "$runtime_repo/scripts/validate_reliable_stage2_parent.py" \
    --parent "$parent" \
    --expected-sha256 "$MECHET_RELIABLE_EXPECTED_PARENT_SHA256" \
    --config "$runtime_repo/configs/agent/natural_language_event_v2_qwen3_0_6b.yaml" \
    --child-config "$config" \
    --project-root "$shared_repo" \
    --manifest "$shared_repo/data/flower_natural_language_event_sft_v2/manifest.json"
fi

python - "$config" "$data_dir" "$stage" "$expected_gpu" "$output" <<'PY'
import hashlib, json, sys, yaml
from pathlib import Path
import torch

config = yaml.safe_load(Path(sys.argv[1]).read_text())
data_dir, stage, expected_gpu, output = Path(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5])
manifest = json.loads((data_dir / 'manifest.json').read_text())
status = json.loads((data_dir / 'ARTIFACT_STATUS.json').read_text())
cache = json.loads((data_dir / 'qwen3_0_6b_tokens_4096/manifest.json').read_text())
expected = {'train': 257167, 'valid': 2890, 'test': 28967}
assert manifest['training_allowed'] is True and status['training_allowed'] is True
for key, filename in (
    ('train_file', 'train.jsonl'), ('validation_file', 'valid.jsonl'),
    ('test_file', 'test.jsonl'),
):
    assert Path(config[key]).resolve() == (data_dir / filename).resolve(), key
assert Path(config['pretokenized_cache_dir']).resolve() == (
    data_dir / 'qwen3_0_6b_tokens_4096'
).resolve()
assert Path(config['output_dir']).resolve() == output.resolve()
if stage == 'state':
    assert not config.get('initial_adapter_path')
assert manifest['reaction_denominator'] == config['contract']['reaction_denominator'] == expected
assert manifest['decision_rows'] == {'train': 2007421, 'valid': 22341, 'test': 225613}
assert cache['model_name_or_path'] == config['model_name_or_path'] == 'Qwen/Qwen3-0.6B'
assert cache['model_revision'] == config['training']['model_revision']
assert cache['max_length'] == config['training']['max_length'] == 4096
assert cache['sources']['train']['sha256'] == manifest['splits']['train']['output_sha256']
assert cache['sources']['validation']['sha256'] == manifest['splits']['valid']['output_sha256']
for split, filename, cache_split in (
    ('train', 'train.jsonl', 'train'), ('valid', 'valid.jsonl', 'validation'),
):
    path = data_dir / filename
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b''):
            digest.update(chunk)
    expected_sha = manifest['splits'][split]['output_sha256']
    assert digest.hexdigest() == expected_sha == cache['sources'][cache_split]['sha256'], split
    print({'gate': 'source_bytes_match_frozen_cache', 'split': split,
           'bytes': path.stat().st_size, 'sha256': expected_sha}, flush=True)
assert cache['splits']['train']['n_rows'] == manifest['decision_rows']['train']
assert cache['splits']['validation']['n_rows'] == manifest['decision_rows']['valid']
assert cache['splits']['train']['truncation_count'] == 0
assert cache['splits']['validation']['truncation_count'] == 0
assert cache['derived_from_identical_tokenizer_cache']['reuse_policy'] == (
    'byte_identical_tokenizer_assets_and_frozen_source_bytes_v1'
)
assert config['training']['num_train_epochs'] == 1.0
assert config['training']['max_steps'] == -1
assert config['limit_examples'] == 0
assert config['contract']['stage'] == ('state_sft' if stage == 'state' else 'trajectory_sft')
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names) == 8 and all(expected_gpu in name.upper() for name in names), names
assert torch.cuda.is_bf16_supported()
print({'gate': 'passed', 'stage': stage, 'reactions': expected,
       'decisions': manifest['decision_rows'], 'gpus': names}, flush=True)
PY

cache_dir=$data_dir/qwen3_0_6b_tokens_4096
local_cache=$(mktemp -d /tmp/mechet_reliable_tokens.XXXXXX)
echo "[reliable-mechet] staging frozen Arrow shards locally"
cp -aL "$cache_dir/." "$local_cache/"
export MECHET_PRETOKENIZED_CACHE_DIR=$local_cache
echo "[reliable-mechet] local cache bytes=$(du -sb "$local_cache" | cut -f1)"

local_hf_cache=$(mktemp -d /tmp/mechet_reliable_hf.XXXXXX)
cp -a "$shared_hf_cache/models--Qwen--Qwen3-0.6B" "$local_hf_cache/"
export HF_HUB_CACHE=$local_hf_cache
echo "[reliable-mechet] pinned model staged locally"

echo "[reliable-mechet] starting one-epoch $stage SFT on 8 ${expected_gpu}s"
torchrun --standalone --nproc_per_node=8 "$runtime_repo/scripts/train_tool_sft.py" \
  --config "$config" ${MECHET_RELIABLE_RESUME:+--resume-from-checkpoint}
test -s "$output/adapter_model.safetensors"
test -s "$output/adapter_manifest.json"
sha256sum "$output/adapter_model.safetensors"
echo "[reliable-mechet] $stage SFT completed"
