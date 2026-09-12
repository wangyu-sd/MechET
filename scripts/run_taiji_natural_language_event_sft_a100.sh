#!/usr/bin/env bash
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=${MECHET_NL_EVENT_RUNTIME_DIR:?set MECHET_NL_EVENT_RUNTIME_DIR}
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
source_dir=$shared_repo/data/flower_inverse_tool_sft_action_delta_v1
output_dir=$shared_repo/data/flower_natural_language_event_sft_v1_rdkit2026
base_config=$runtime_repo/configs/agent/natural_language_event_sft_qwen3_8b_a100.yaml
liger_wheel=$shared_repo/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl
xformers_wheel=$shared_repo/artifacts/wheels/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl
bitsandbytes_wheel=$shared_repo/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl
rdkit_wheel=$shared_repo/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared_repo"

echo "[natural-language-sft] runtime=$runtime_repo source=$source_dir output=$output_dir"
test -f "$base_config"
test -f "$source_dir/train.jsonl"
echo "303b9bbf5c10f9289c3139afb41e4d989e8c809516624a106b89b064163d971d  $liger_wheel" | sha256sum --check --strict
echo "bbf2f500dfdbcf4649bf568cc2c9f434399f704dc4064fd1fbdbef2b524a8139  $xformers_wheel" | sha256sum --check --strict
echo "54b771f06e1a3c73af5c7f16ccf0fc23a846052813d4b008d10cb6e017dd1c8c  $bitsandbytes_wheel" | sha256sum --check --strict
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $rdkit_wheel" | sha256sum --check --strict

runtime_target=$(mktemp -d /tmp/mechet_nl_event_runtime.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" \
  "$liger_wheel" "$xformers_wheel" "$bitsandbytes_wheel" "$rdkit_wheel"

export HF_HUB_CACHE=$shared_hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$runtime_target:$runtime_repo/src:$runtime_repo
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

python - <<'PY'
import os, rdkit, torch
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, got {torch.cuda.device_count()}")
names=[torch.cuda.get_device_name(i) for i in range(8)]
if not all("A100" in name.upper() for name in names):
    raise SystemExit(f"expected A100 GPUs, got {names}")
if rdkit.__version__ != "2026.03.4":
    raise SystemExit(f"expected RDKit 2026.03.4, got {rdkit.__version__}")
print({'gpus':names,'cpu_affinity':len(os.sched_getaffinity(0)),
       'rdkit_version':rdkit.__version__},flush=True)
PY

manifest=$output_dir/manifest.json
if ! python - "$manifest" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file(): raise SystemExit(1)
m=json.loads(p.read_text())
ok=m.get('training_allowed') is True and m.get('status')=='validated_complete'
ok &= m.get('reaction_denominator')=={'train':257167,'valid':2890,'test':28967}
ok &= m.get('reference_replay_verified') is True
ok &= m.get('model_visible_atom_maps') is False and m.get('model_predicts_state') is False
raise SystemExit(0 if ok else 1)
PY
then
  workers=$(python - <<'PY'
import os
print(min(64, len(os.sched_getaffinity(0))))
PY
)
  echo "[natural-language-sft] full CPU conversion starts workers=$workers"
  python -u "$runtime_repo/scripts/build_natural_language_event_sft.py" \
    --source-dir "$source_dir" --output-dir "$output_dir" --workers "$workers"
fi

generated_config=$(mktemp /tmp/mechet_nl_event_config.XXXXXX.yaml)
python - "$base_config" "$manifest" "$generated_config" <<'PY'
import json,sys,yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text())
m=json.loads(Path(sys.argv[2]).read_text())
assert m['training_allowed'] is True and m['reference_replay_verified'] is True
assert m['reaction_denominator']=={'train':257167,'valid':2890,'test':28967}
cfg['contract']['expected_train_rows']=m['decision_rows']['train']
cfg['contract']['expected_validation_rows']=m['decision_rows']['valid']
cfg['contract']['expected_test_rows']=m['decision_rows']['test']
Path(sys.argv[3]).write_text(yaml.safe_dump(cfg,sort_keys=False))
print({'dataset_gate':'passed','reactions':m['reaction_denominator'],
       'decisions':m['decision_rows']},flush=True)
PY

cache_manifest=$output_dir/qwen3_8b_tokens_4096/manifest.json
if [[ ! -f "$cache_manifest" ]]; then
  echo "[natural-language-sft] distributed tokenization starts"
  torchrun --standalone --nproc_per_node=8 \
    "$runtime_repo/scripts/prepare_tool_sft_arrow.py" --config "$generated_config"
fi

python - "$cache_manifest" "$manifest" <<'PY'
import json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text()); m=json.loads(Path(sys.argv[2]).read_text())
assert c['splits']['train']['n_rows']==m['decision_rows']['train']
assert c['splits']['validation']['n_rows']==m['decision_rows']['valid']
assert c['splits']['train']['truncation_count']==0
assert c['splits']['validation']['truncation_count']==0
print({'token_gate':'passed','train':{k:c['splits']['train'][k] for k in
      ['n_rows','n_training_windows','n_windowed_source_rows','total_input_tokens',
       'total_supervised_tokens','max_raw_input_tokens','p95_raw_input_tokens']}},flush=True)
PY

local_cache=$(mktemp -d /tmp/mechet_nl_event_tokens.XXXXXX)
echo "[natural-language-sft] staging token cache: $local_cache"
cp -a "$output_dir/qwen3_8b_tokens_4096/." "$local_cache/"
export MECHET_PRETOKENIZED_CACHE_DIR=$local_cache

local_hf_cache=$(mktemp -d /tmp/mechet_nl_event_hf.XXXXXX)
echo "[natural-language-sft] staging pinned model cache: $local_hf_cache"
cp -a "$shared_hf_cache/models--Qwen--Qwen3-8B" "$local_hf_cache/"
export HF_HUB_CACHE=$local_hf_cache

echo "[natural-language-sft] one-epoch Qwen3-8B training starts"
exec torchrun --standalone --nproc_per_node=8 \
  "$runtime_repo/scripts/train_tool_sft.py" --config "$generated_config"
