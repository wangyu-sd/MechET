#!/usr/bin/env bash
set -Eeuo pipefail

repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr81-system-one-20261004
shared=/aaa/fionafyang/buddy1/whaleywang/MechET
strict_data="$shared/data/mech_uspto_31k_natural_language_history_v2"
full_data="${MECHET_FULL_ENDPOINT_SOURCE_DIR:-$shared/data/mech_uspto_31k_full_endpoint_rxnmapper}"
route_checkpoint="$shared/outputs/agent/system_one_pr81_phase0_31k_20261005_v2/full"
route_run="$shared/outputs/agent/system_one_pr81_phase1a_route_31k_20261005"
typed_checkpoint="$shared/outputs/agent/system_one_pr81_jev_typed_v2_31k_20261005/full"
split="${MECHET_FULL_ENDPOINT_SPLIT:-valid}"
case "$split" in
  valid|test) ;;
  *) printf 'invalid MECHET_FULL_ENDPOINT_SPLIT=%s\n' "$split" >&2; exit 2 ;;
esac
context_run="${MECHET_FULL_ENDPOINT_CONTEXT_RUN:-$shared/outputs/agent/system_one_pr81_full_context_knn_${split}_20261005}"
output="${MECHET_FULL_ENDPOINT_OUTPUT:-$shared/outputs/agent/system_one_pr81_full_endpoint_${split}64_20261005}"
limit="${MECHET_FULL_ENDPOINT_LIMIT:-64}"
expected="${MECHET_FULL_ENDPOINT_EXPECTED:-64}"
expected_field="${MECHET_FULL_ENDPOINT_PRODUCT_FIELD:-rxn_prod_min}"
log_every="${MECHET_FULL_ENDPOINT_LOG_EVERY:-8}"
target_focus="${MECHET_FULL_ENDPOINT_FIRST_EVENT_TARGET_FOCUS:-0}"
principal_prompt="${MECHET_FULL_ENDPOINT_PRINCIPAL_TARGET_PROMPT:-0}"
case "$target_focus" in
  0|1) ;;
  *) printf 'invalid MECHET_FULL_ENDPOINT_FIRST_EVENT_TARGET_FOCUS=%s\n' "$target_focus" >&2; exit 2 ;;
esac
case "$principal_prompt" in
  0|1) ;;
  *) printf 'invalid MECHET_FULL_ENDPOINT_PRINCIPAL_TARGET_PROMPT=%s\n' "$principal_prompt" >&2; exit 2 ;;
esac
shared_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$repo/src:$repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wheels=$(mktemp -d /tmp/mechet_system_one_full_endpoint_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheels" \
  "$shared/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
export PYTHONPATH="$wheels:$PYTHONPATH"
python - <<'PY'
import rdkit
import torch
print({'phase': 'runtime_gate', 'rdkit': rdkit.__version__,
       'gpu_count': torch.cuda.device_count(),
       'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
      flush=True)
assert rdkit.__version__ == '2026.03.4'
assert torch.cuda.device_count() == 1
assert 'A100' in torch.cuda.get_device_name(0).upper()
PY

test -s "$full_data/manifest.json"
test -s "$context_run/report.json"
test -s "$context_run/cases.jsonl"
test -s "$route_checkpoint/run_manifest_epoch1.json"
test -s "$route_run/action_family_head.pt"
test -s "$typed_checkpoint/run_manifest_epoch1.json"
test ! -e "$output"

model_cache=$(mktemp -d /tmp/mechet_system_one_full_endpoint_hf.XXXXXX)
printf '[system-one-full-endpoint] staging pinned model\n'
cp -a "$shared_cache/models--Qwen--Qwen3-0.6B" "$model_cache/"
export HF_HUB_CACHE="$model_cache"

printf '[system-one-full-endpoint] starting %s limit=%s expected=%s\n' "$split" "$limit" "$expected"
extra_args=()
if [[ "$target_focus" == 1 ]]; then
  extra_args+=(--first-event-target-focus)
fi
if [[ "$principal_prompt" == 1 ]]; then
  extra_args+=(--principal-target-prompt)
fi
python scripts/eval_system_one_full_endpoint.py \
  --full-endpoint-dir "$full_data" \
  --strict-dir "$strict_data" \
  --context-run "$context_run" \
  --split "$split" \
  --route-checkpoint "$route_checkpoint" \
  --route-run "$route_run" \
  --typed-checkpoint "$typed_checkpoint" \
  --output "$output" \
  --limit "$limit" \
  --seed 17 \
  --max-actions 12 \
  --log-every "$log_every" \
  "${extra_args[@]}"
test -s "$output/report.json"
test -s "$output/cases.jsonl"
python - "$output/report.json" "$output/cases.jsonl" "$split" "$expected" "$expected_field" "$target_focus" "$principal_prompt" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
cases = sum(1 for line in Path(sys.argv[2]).open() if line.strip())
assert report['split'] == sys.argv[3]
assert report['full_endpoint_reaction_denominator'] == 3120
assert report['evaluated_reactions'] == cases == int(sys.argv[4])
assert report['product_source_field'] == sys.argv[5]
assert report['first_event_target_focus'] == (sys.argv[6] == '1')
assert report['principal_target_prompt'] == (sys.argv[7] == '1')
assert report['input_contract'] == 'principal_product_only_with_train_only_predicted_context'
assert report['output_contract'] == 'full_endpoint_structural_precursor_product_origin_projection'
print({'phase': 'validated_full_endpoint_report', 'evaluated_reactions': cases,
       'structural_exact': report['structural_exact']}, flush=True)
PY
