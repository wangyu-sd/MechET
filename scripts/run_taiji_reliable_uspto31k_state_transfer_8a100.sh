#!/usr/bin/env bash
# Product-only, zero-shot FlowER State-SFT transfer to the complete 1,253-row
# current-compiler executable mech-USPTO-31k trace-view test.
set -Eeuo pipefail

shared=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime=/aaa/fionafyang/buddy1/whaleywang/MechET-pr82-reliable-20261006
hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
source_data=$shared/data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/test.jsonl
source_manifest=$shared/data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/manifest.json
decisions=$shared/data/mech_uspto_31k_natural_language_event_v2/test.jsonl
decision_manifest=$shared/data/mech_uspto_31k_natural_language_event_v2/manifest.json
adapter=$shared/outputs/agent/natural_language_event_v2_qwen3_0_6b_seed17
expected_adapter_sha=fbd8db06094fd8029d4cb0cac38cbb88280a442d76f686258194723019b1a1bd
search_mode=${MECHET_SEARCH_MODE:-k1}
case "$search_mode" in
  k1)
    output=$shared/outputs/eval/reliable_mechet_state_uspto31k_trace1253_k1_seed17_20261007
    search_args=(--branching 1 --early-beam 1 --late-beam 1)
    audit_args=()
    ;;
  beam)
    output=$shared/outputs/eval/reliable_mechet_state_uspto31k_trace1253_beam4_seed17_20261008
    search_args=(--beam-search --branching 4 --early-beam 4 --late-beam 4 --early-depth 40)
    audit_args=(--beam-search)
    ;;
  *) echo "unknown MECHET_SEARCH_MODE: $search_mode" >&2; exit 2 ;;
esac

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$shared"
mountpoint -q /aaa/fionafyang/buddy1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_CACHE=$hf_cache
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=$runtime/src:$runtime

if [[ -e $output ]]; then
  echo "refusing to overwrite existing evaluation: $output" >&2
  exit 3
fi

rdkit_wheel=$shared/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $rdkit_wheel" | sha256sum --check --strict
rdkit_target=$(mktemp -d /tmp/mechet_31k_state_transfer_rdkit.XXXXXX)
python -m pip install --quiet --no-deps --target "$rdkit_target" "$rdkit_wheel"
export PYTHONPATH=$rdkit_target:$PYTHONPATH

python - "$source_data" "$source_manifest" "$decisions" "$decision_manifest" "$adapter" "$expected_adapter_sha" <<'PY'
import hashlib, json, sys
from pathlib import Path
import rdkit, torch
from scripts.audit_reliable_strict_test_eval import validate_source
from scripts.run_natural_language_value_search import (
    policy_prompt, product_only_private_state, validate_v2_adapter_manifest, visible,
)
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS

source, manifest, decisions, decision_manifest, adapter = map(Path, sys.argv[1:6])
expected_sha = sys.argv[6]
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names) == 8 and all('A100' in name.upper() for name in names), names
assert rdkit.__version__ == '2026.03.4', rdkit.__version__
ids, source_sha = validate_source(
    source=source, manifest=manifest, expected_rows=1253,
    check_product_only_mapping=True,
)
source_status = json.loads((source.parent / 'ARTIFACT_STATUS.json').read_text())
decision_status = json.loads((decisions.parent / 'ARTIFACT_STATUS.json').read_text())
frozen_decisions = json.loads(decision_manifest.read_text())
assert source_status['status'] == 'validated' and source_status['rows']['test'] == 1253
assert decision_status['status'] == 'validated_trace_view'
assert frozen_decisions['reaction_denominator'] == {'train': 10152, 'valid': 1319, 'test': 1253}
assert frozen_decisions['full_reaction_denominator']['test'] == 3120
assert frozen_decisions['splits']['test']['source_sha256'] == source_sha
digest = hashlib.sha256(decisions.read_bytes()).hexdigest()
assert digest == frozen_decisions['splits']['test']['output_sha256']
validate_v2_adapter_manifest(
    adapter, compact_history=False, expected_model='Qwen/Qwen3-0.6B',
    expected_revision='c1899de289a04d12100db370d81485cdf75e47ca',
)
assert hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest() == expected_sha
first = {}
with decisions.open(encoding='utf-8') as stream:
    for line in stream:
        row = json.loads(line)
        key = str(row['source_id'])
        if key not in first:
            assert int(row['metadata']['decision_index']) == 0, key
            first[key] = row
assert len(first) == len(ids) == 1253
count = 0
with source.open(encoding='utf-8') as stream:
    for line in stream:
        row = json.loads(line)
        key = str(row['source_id'])
        state = product_only_private_state(str(row['target_smiles']))
        prompt = policy_prompt(visible(state), state, include_inventory=True,
                               actions=[], compact_history=False)
        frozen = first[key]
        assert frozen['messages'][0]['content'] == SYSTEM and frozen['tools'] == TOOLS, key
        assert prompt == frozen['messages'][1]['content'], key
        count += 1
print({'gate': 'passed', 'model': 'Qwen3-0.6B FlowER State-SFT',
       'test_rows': count, 'first_prompt_exact': count,
       'source_sha256': source_sha, 'adapter_sha256': expected_sha,
       'rdkit': rdkit.__version__, 'gpus': names}, flush=True)
PY

echo "[meteor-31k-transfer] starting product-only $search_mode, 1,253 reactions, 8 A100 workers"
torchrun --standalone --nproc_per_node=8 \
  "$runtime/scripts/run_natural_language_value_search.py" \
  --data "$source_data" --output "$output" \
  --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --policy-adapter "$adapter" --sample-reactions 1253 --seed 17 \
  --matched-v2 --product-only-remap --reject-target-retained-finish \
  "${search_args[@]}" \
  --max-decisions 40 --max-imports 32 --max-new-tokens 512 \
  --value-weight 0 --no-4bit
python -u "$runtime/scripts/summarize_natural_language_value_search.py" \
  --search-dir "$output"
python -u "$runtime/scripts/audit_reliable_uspto31k_transfer.py" \
  --source "$source_data" --manifest "$source_manifest" \
  --results-dir "$output" --adapter "$adapter" \
  --expected-adapter-sha256 "$expected_adapter_sha" \
  "${audit_args[@]}" \
  --output "$output/transfer_audit.json"
echo '[meteor-31k-transfer] complete; raw decision trajectories and audit saved'
