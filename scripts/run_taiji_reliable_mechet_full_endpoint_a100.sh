#!/usr/bin/env bash
# Full unfiltered FlowER product-only endpoint test; separate from strict process view.
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr82-reliable-20261006
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
source_data=$shared_repo/data/flower_full_endpoint_sft/test.jsonl
source_manifest=$shared_repo/data/flower_full_endpoint_sft/manifest.json
stage=${MECHET_RELIABLE_STAGE:?set MECHET_RELIABLE_STAGE=state or trajectory}
case "$stage" in
  state)
    adapter=$shared_repo/outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17
    history_flag=()
    ;;
  trajectory)
    adapter=$shared_repo/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17
    history_flag=(--compact-history)
    ;;
  *) echo "unsupported stage=$stage" >&2; exit 2 ;;
esac
output=${MECHET_RELIABLE_EVAL_OUTPUT:-$shared_repo/outputs/eval/reliable_mechet_${stage}_flower_full_endpoint_k1_seed17}
if [[ $output != /* ]]; then
  echo "MECHET_RELIABLE_EVAL_OUTPUT must be an absolute path" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite an existing full-test evaluation: $output" >&2
  exit 3
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared_repo"
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$shared_hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$runtime_repo/src:$runtime_repo

rdkit_wheel=$shared_repo/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $rdkit_wheel" | sha256sum --check --strict
runtime_target=$(mktemp -d /tmp/mechet_reliable_endpoint_rdkit.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" "$rdkit_wheel"
export PYTHONPATH=$runtime_target:$PYTHONPATH

python - "$source_data" "$source_manifest" "$adapter" "$stage" <<'PY'
import sys
from pathlib import Path
import rdkit
import torch
from scripts.audit_reliable_full_endpoint_eval import validate_source
from scripts.run_natural_language_value_search import validate_v2_adapter_manifest

source, manifest, adapter = map(Path, sys.argv[1:4])
stage = sys.argv[4]
ids, digest = validate_source(
    source=source, manifest=manifest, check_product_only_mapping=True,
)
validate_v2_adapter_manifest(
    adapter, compact_history=stage == 'trajectory',
    expected_model='Qwen/Qwen3-0.6B',
    expected_revision='c1899de289a04d12100db370d81485cdf75e47ca',
)
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names) == 8 and all('A100' in name.upper() for name in names), names
assert rdkit.__version__ == '2026.03.4'
print({'gate': 'full_endpoint_source_and_adapter_valid', 'stage': stage,
       'rows': len(ids), 'source_sha256': digest, 'gpus': names}, flush=True)
PY

adapter_sha=$(sha256sum "$adapter/adapter_model.safetensors" | cut -d' ' -f1)
echo "[reliable-full-endpoint] frozen adapter_sha256=$adapter_sha"
echo "[reliable-full-endpoint] stage=$stage K=1 product-only test=28971"
torchrun --standalone --nproc_per_node=8 \
  "$runtime_repo/scripts/run_natural_language_value_search.py" \
  --data "$source_data" --output "$output" \
  --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --policy-adapter "$adapter" --sample-reactions 28971 --seed 17 \
  --matched-v2 --product-only-remap "${history_flag[@]}" \
  --reject-target-retained-finish \
  --branching 1 --early-beam 1 --late-beam 1 \
  --max-decisions 40 --max-imports 32 --max-new-tokens 512 \
  --value-weight 0 --no-4bit
python -u "$runtime_repo/scripts/audit_reliable_full_endpoint_eval.py" \
  --source "$source_data" --manifest "$source_manifest" \
  --results-dir "$output" --output "$output/full_endpoint_audit.json" \
  --adapter "$adapter" --expected-adapter-sha256 "$adapter_sha" --stage "$stage"
