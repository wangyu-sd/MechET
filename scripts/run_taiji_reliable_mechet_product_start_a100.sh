#!/usr/bin/env bash
# Matched product-start K=1 validation diagnostic for PR #82 Stage I/II.
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr82-reliable-20261006
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
stage=${MECHET_RELIABLE_STAGE:?set MECHET_RELIABLE_STAGE=state or trajectory}
case "$stage" in
  state)
    adapter=$shared_repo/outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17
    output=$shared_repo/outputs/eval/reliable_mechet_state_valid128_product_start_seed17
    decision_data=$shared_repo/data/flower_natural_language_event_sft_v2/valid.jsonl
    history_flag=()
    ;;
  trajectory)
    adapter=$shared_repo/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17
    output=$shared_repo/outputs/eval/reliable_mechet_trajectory_valid128_product_start_seed17
    decision_data=$shared_repo/data/flower_natural_language_event_history_v2/valid.jsonl
    history_flag=(--compact-history)
    ;;
  *) echo "invalid MECHET_RELIABLE_STAGE=$stage" >&2; exit 2 ;;
esac
source_data=$shared_repo/data/flower_inverse_tool_sft_action_delta_v1/valid.jsonl

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
test -s "$decision_data"
test -s "$adapter/adapter_model.safetensors"
test -s "$adapter/adapter_manifest.json"
echo "[reliable-product-start] stage=$stage adapter_sha256=$(sha256sum "$adapter/adapter_model.safetensors" | cut -d' ' -f1)"

rdkit_wheel=$shared_repo/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $rdkit_wheel" | sha256sum --check --strict
runtime_target=$(mktemp -d /tmp/mechet_reliable_product_start.XXXXXX)
python -m pip install --quiet --no-deps --target "$runtime_target" "$rdkit_wheel"
export PYTHONPATH=$runtime_target:$PYTHONPATH

python - "$adapter" "$stage" <<'PY'
import sys, torch, rdkit
from pathlib import Path
from scripts.run_natural_language_value_search import validate_v2_adapter_manifest

adapter, stage = Path(sys.argv[1]), sys.argv[2]
validate_v2_adapter_manifest(
    adapter, compact_history=stage == 'trajectory',
    expected_model='Qwen/Qwen3-0.6B',
    expected_revision='c1899de289a04d12100db370d81485cdf75e47ca',
)
assert torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0).upper()
assert rdkit.__version__ == '2026.03.4'
print({'gate':'passed','stage':stage,'gpu':torch.cuda.get_device_name(0)}, flush=True)
PY

python - "$source_data" "$decision_data" "$stage" <<'PY'
import json, sys
from pathlib import Path
from scripts.run_natural_language_value_search import (
    policy_prompt, product_only_private_state, read_selected, visible,
)

source, decisions, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows = read_selected(source, 128, 17)
selected = {str(row['source_id']) for row in rows}
first = {}
with decisions.open(encoding='utf-8') as stream:
    for line in stream:
        decision = json.loads(line)
        reaction = str(decision['source_id'])
        if reaction in selected and reaction not in first:
            if int(decision['metadata']['decision_index']) != 0:
                raise ValueError(f'{reaction}: first supervised decision is not product-start')
            first[reaction] = decision
if set(first) != selected:
    raise ValueError('128 product-start decisions do not align to the source IDs')
for row in rows:
    reaction = str(row['source_id'])
    state = product_only_private_state(str(row['target_smiles']))
    prompt = policy_prompt(
        visible(state), state, include_inventory=True, actions=[],
        compact_history=stage == 'trajectory',
    )
    if prompt != first[reaction]['messages'][1]['content']:
        raise ValueError(f'{reaction}: product-start prompt differs from frozen SFT')
print({'gate':'128_product_start_prompts_match_frozen_sft','stage':stage}, flush=True)
PY

echo "[reliable-product-start] product-only K=1 validation, 128 fixed IDs, 40-decision budget"
torchrun --standalone --nproc_per_node=1 \
  "$runtime_repo/scripts/run_natural_language_value_search.py" \
  --data "$source_data" --output "$output" \
  --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --policy-adapter "$adapter" --sample-reactions 128 --seed 17 \
  --matched-v2 --product-only-remap "${history_flag[@]}" \
  --reject-target-retained-finish \
  --branching 1 --early-beam 1 --late-beam 1 \
  --max-decisions 40 --max-imports 32 --max-new-tokens 512 \
  --value-weight 0 --no-4bit
python -u "$runtime_repo/scripts/summarize_natural_language_value_search.py" \
  --search-dir "$output"
python - "$source_data" "$output" <<'PY'
import json, sys
from pathlib import Path
from scripts.run_natural_language_value_search import read_selected

source, output = Path(sys.argv[1]), Path(sys.argv[2])
expected = {str(row['id']) for row in read_selected(source, 128, 17)}
paths = sorted(output.glob('results.shard-*.jsonl'))
rows = [json.loads(line) for path in paths for line in path.open() if line.strip()]
actual = [str(row['id']) for row in rows]
if len(rows) != 128 or len(set(actual)) != 128 or set(actual) != expected:
    raise ValueError('product-start evaluation did not cover exactly the frozen 128 IDs')
print({'gate':'fixed_128_ids_complete','report':str(output / 'evaluation.json')}, flush=True)
PY
python -u "$runtime_repo/scripts/analyze_reliable_product_start.py" \
  --source "$source_data" --decisions "$decision_data" \
  --results "$output/results.shard-00-of-01.jsonl" --output "$output" \
  --sample-reactions 128 --seed 17
