#!/usr/bin/env bash
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=${MECHET_VALUE_RUNTIME_DIR:?set MECHET_VALUE_RUNTIME_DIR}
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
source_dir=$shared_repo/data/flower_inverse_tool_sft_action_delta_v1
value_data=$shared_repo/data/flower_natural_language_state_value_v1
distill_data=$shared_repo/data/flower_natural_language_value_distill_v1
policy_adapter=$shared_repo/outputs/agent/natural_language_event_sft_qwen3_8b_a100_seed17_20260913
value_adapter=$shared_repo/outputs/agent/natural_language_state_value_qwen3_8b_h20_seed17_20260915
distill_adapter=$shared_repo/outputs/agent/natural_language_value_distill_qwen3_8b_h20_seed17_20260915
search_root=$shared_repo/outputs/eval/natural_language_value_pipeline_20260915
value_config=$runtime_repo/configs/agent/natural_language_state_value_qwen3_8b_h20.yaml
distill_config=$runtime_repo/configs/agent/natural_language_value_distill_qwen3_8b_h20.yaml
liger_wheel=$shared_repo/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl
xformers_wheel=$shared_repo/artifacts/wheels/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl
bitsandbytes_wheel=$shared_repo/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl
rdkit_wheel=$shared_repo/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared_repo"

echo "[meteor-value-pipeline] runtime=$runtime_repo"
echo "[meteor-value-pipeline] stages=data,value-SFT,product-only-search,distill-SFT,product-only-eval"
test -f "$source_dir/train.jsonl"
test -f "$policy_adapter/adapter_config.json"
test -f "$value_config"
test -f "$distill_config"
echo "303b9bbf5c10f9289c3139afb41e4d989e8c809516624a106b89b064163d971d  $liger_wheel" | sha256sum --check --strict
echo "bbf2f500dfdbcf4649bf568cc2c9f434399f704dc4064fd1fbdbef2b524a8139  $xformers_wheel" | sha256sum --check --strict
echo "54b771f06e1a3c73af5c7f16ccf0fc23a846052813d4b008d10cb6e017dd1c8c  $bitsandbytes_wheel" | sha256sum --check --strict
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $rdkit_wheel" | sha256sum --check --strict

runtime_target=$(mktemp -d /tmp/mechet_value_runtime.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" \
  "$liger_wheel" "$xformers_wheel" "$bitsandbytes_wheel" "$rdkit_wheel"

local_hf_cache=$(mktemp -d /tmp/mechet_value_hf.XXXXXX)
cp -a "$shared_hf_cache/models--Qwen--Qwen3-8B" "$local_hf_cache/"
export HF_HUB_CACHE=$local_hf_cache
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
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names)==8, names
assert all('H20' in value.upper() for value in names), names
assert rdkit.__version__=='2026.03.4', rdkit.__version__
print({'stage':'hardware-gate','gpus':names,'cpus':len(os.sched_getaffinity(0)),
       'rdkit':rdkit.__version__},flush=True)
PY

mkdir -p "$search_root"

if ! python - "$value_data/manifest.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file(): raise SystemExit(1)
m=json.loads(p.read_text())
ok=m.get('training_allowed') is True
ok &= m.get('reports',{}).get('train',{}).get('selected_reactions')==20000
ok &= m.get('reports',{}).get('valid',{}).get('selected_reactions')==512
ok &= m.get('reports',{}).get('train',{}).get('label_C',0)>0
raise SystemExit(0 if ok else 1)
PY
then
  echo "[meteor-value-pipeline] stage 1/6 build value supervision"
  workers=$(python - <<'PY'
import os
print(min(64,len(os.sched_getaffinity(0))))
PY
)
  rm -rf "$value_data"
  python -u "$runtime_repo/scripts/build_natural_language_state_value.py" \
    --source-dir "$source_dir" --output-dir "$value_data" \
    --train-reactions 20000 --valid-reactions 512 --workers "$workers"
fi

if [[ ! -f "$value_adapter/adapter_model.safetensors" ]]; then
  echo "[meteor-value-pipeline] stage 2/6 tokenize and train state-value critic"
  rm -rf "$value_data/qwen3_8b_tokens_1024"
  torchrun --standalone --nproc_per_node=8 \
    "$runtime_repo/scripts/prepare_tool_sft_arrow.py" --config "$value_config"
  local_value_cache=$(mktemp -d /tmp/mechet_value_tokens.XXXXXX)
  cp -a "$value_data/qwen3_8b_tokens_1024/." "$local_value_cache/"
  MECHET_PRETOKENIZED_CACHE_DIR=$local_value_cache \
    torchrun --standalone --nproc_per_node=8 \
      "$runtime_repo/scripts/train_tool_sft.py" --config "$value_config"
fi

echo "[meteor-value-pipeline] stage 3/6 pre-distillation product-only validation"
rm -rf "$search_root/pre_distill_valid128"
torchrun --standalone --nproc_per_node=8 \
  "$runtime_repo/scripts/run_natural_language_value_search.py" \
  --data "$source_dir/valid.jsonl" --output "$search_root/pre_distill_valid128" \
  --policy-adapter "$policy_adapter" --value-adapter "$value_adapter" \
  --sample-reactions 128 --branching 4 --early-beam 4 --late-beam 2
python -u "$runtime_repo/scripts/summarize_natural_language_value_search.py" \
  --search-dir "$search_root/pre_distill_valid128"

if [[ ! -f "$distill_adapter/adapter_model.safetensors" ]]; then
  echo "[meteor-value-pipeline] stage 4/6 product-only train search and verified distillation data"
  rm -rf "$search_root/train256" "$distill_data"
  torchrun --standalone --nproc_per_node=8 \
    "$runtime_repo/scripts/run_natural_language_value_search.py" \
    --data "$source_dir/train.jsonl" --output "$search_root/train256" \
    --policy-adapter "$policy_adapter" --value-adapter "$value_adapter" \
    --sample-reactions 256 --branching 4 --early-beam 4 --late-beam 2 --write-distill
  python -u "$runtime_repo/scripts/summarize_natural_language_value_search.py" \
    --search-dir "$search_root/train256" --distill-dir "$distill_data" \
    --minimum-successes 16

  echo "[meteor-value-pipeline] stage 5/6 distill successful executable paths"
  rm -rf "$distill_data/qwen3_8b_tokens_4096"
  torchrun --standalone --nproc_per_node=8 \
    "$runtime_repo/scripts/prepare_tool_sft_arrow.py" --config "$distill_config"
  local_distill_cache=$(mktemp -d /tmp/mechet_distill_tokens.XXXXXX)
  cp -a "$distill_data/qwen3_8b_tokens_4096/." "$local_distill_cache/"
  MECHET_PRETOKENIZED_CACHE_DIR=$local_distill_cache \
    torchrun --standalone --nproc_per_node=8 \
      "$runtime_repo/scripts/train_tool_sft.py" --config "$distill_config"
fi

echo "[meteor-value-pipeline] stage 6/6 post-distillation product-only validation"
rm -rf "$search_root/post_distill_valid128"
torchrun --standalone --nproc_per_node=8 \
  "$runtime_repo/scripts/run_natural_language_value_search.py" \
  --data "$source_dir/valid.jsonl" --output "$search_root/post_distill_valid128" \
  --policy-adapter "$distill_adapter" --value-adapter "$value_adapter" \
  --sample-reactions 128 --branching 4 --early-beam 4 --late-beam 2
python -u "$runtime_repo/scripts/summarize_natural_language_value_search.py" \
  --search-dir "$search_root/post_distill_valid128"

python - "$search_root" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
pre=json.loads((root/'pre_distill_valid128/evaluation.json').read_text())
post=json.loads((root/'post_distill_valid128/evaluation.json').read_text())
report={'status':'complete','protocol':'product_only_value_guided_search_v1',
        'reference_endpoint_model_visible':False,'pre_distill':pre,'post_distill':post,
        'delta_top1':post['top1_accuracy']-pre['top1_accuracy'],
        'delta_pass_at_beam':post['pass_at_beam_accuracy']-pre['pass_at_beam_accuracy']}
(root/'FINAL_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
print(report,flush=True)
PY
echo "[meteor-value-pipeline] complete report=$search_root/FINAL_REPORT.json"
