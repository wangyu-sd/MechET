#!/usr/bin/env bash
# Frozen, teacher-forced-state local diagnostic; never report as product-start accuracy.
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr82-reliable-20261006
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
stage=${MECHET_RELIABLE_STAGE:?set MECHET_RELIABLE_STAGE=state or trajectory}
source_data=$shared_repo/data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl
case "$stage" in
  state)
    decision_dir=$shared_repo/data/flower_natural_language_event_sft_v2
    adapter=$shared_repo/outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17
    output=$shared_repo/outputs/eval/reliable_mechet_state_valid128_local_seed17
    ;;
  trajectory)
    decision_dir=$shared_repo/data/flower_natural_language_event_history_v2
    adapter=$shared_repo/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17
    output=$shared_repo/outputs/eval/reliable_mechet_trajectory_valid128_local_seed17
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

test -s "$source_data"
test -s "$decision_dir/valid.jsonl"
test -s "$decision_dir/manifest.json"
test -s "$adapter/adapter_model.safetensors"
test -s "$adapter/adapter_manifest.json"
echo "[reliable-local] stage=$stage adapter_sha256=$(sha256sum "$adapter/adapter_model.safetensors" | cut -d' ' -f1)"

rdkit_wheel=$shared_repo/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $rdkit_wheel" | sha256sum --check --strict
runtime_target=$(mktemp -d /tmp/mechet_reliable_eval.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" "$rdkit_wheel"
export PYTHONPATH=$runtime_target:$PYTHONPATH

python - "$adapter/adapter_manifest.json" "$decision_dir/manifest.json" <<'PY'
import json, sys, torch
from pathlib import Path
import rdkit
adapter = json.loads(Path(sys.argv[1]).read_text())
data = json.loads(Path(sys.argv[2]).read_text())
assert adapter['base_model'] == 'Qwen/Qwen3-0.6B'
assert adapter['base_model_revision'] == 'c1899de289a04d12100db370d81485cdf75e47ca'
assert data['reaction_denominator'] == {'train':257167,'valid':2890,'test':28967}
assert data['training_allowed'] is True
assert torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0).upper()
assert rdkit.__version__ == '2026.03.4'
print({'gate':'passed','gpu':torch.cuda.get_device_name(0),'rdkit':rdkit.__version__},flush=True)
PY

common=(--data "$source_data" --decision-data "$decision_dir/valid.jsonl"
  --output "$output" --model Qwen/Qwen3-0.6B
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca
  --adapter "$adapter" --sample-reactions 128 --seed 17
  --no-4bit --dtype bfloat16 --sft-aligned-prefix --import-role-breakdown)
echo "[reliable-local] generating 128-reaction matched local decisions"
torchrun --standalone --nproc_per_node=1 \
  "$runtime_repo/scripts/eval_natural_language_event_local.py" run "${common[@]}" \
  --batch-size 4 --max-new-tokens 512 --max-context 4096
python -u "$runtime_repo/scripts/eval_natural_language_event_local.py" aggregate "${common[@]}"
echo "[reliable-local] complete report=$output/evaluation.json"
