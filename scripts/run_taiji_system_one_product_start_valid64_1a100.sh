#!/usr/bin/env bash
set -Eeuo pipefail

repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr81-system-one-20261004
shared=/aaa/fionafyang/buddy1/whaleywang/MechET
source_data="$shared/data/mech_uspto_31k_natural_language_history_v2"
route_checkpoint="$shared/outputs/agent/system_one_pr81_phase0_31k_20261005_v2/full"
route_run="$shared/outputs/agent/system_one_pr81_phase1a_route_31k_20261005"
typed_checkpoint="$shared/outputs/agent/system_one_pr81_jev_typed_v2_31k_20261005/full"
output="${MECHET_PRODUCT_START_OUTPUT:-$shared/outputs/agent/system_one_pr81_product_start_valid64_20261005}"
limit="${MECHET_PRODUCT_START_LIMIT:-64}"
expected="${MECHET_PRODUCT_START_EXPECTED:-64}"
log_every="${MECHET_PRODUCT_START_LOG_EVERY:-8}"
legality_backoff="${MECHET_PRODUCT_START_LEGALITY_BACKOFF:-0}"
case "$legality_backoff" in
  0) backoff_args=() ;;
  1) backoff_args=(--legality-backoff) ;;
  *) printf 'invalid MECHET_PRODUCT_START_LEGALITY_BACKOFF=%s\n' "$legality_backoff" >&2; exit 2 ;;
esac
shared_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$repo"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$repo/src:$repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wheels=$(mktemp -d /tmp/mechet_system_one_product_wheels.XXXXXX)
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

test -s "$route_checkpoint/run_manifest_epoch1.json"
test -s "$route_run/action_family_head.pt"
test -s "$typed_checkpoint/run_manifest_epoch1.json"
test ! -e "$output"

model_cache=$(mktemp -d /tmp/mechet_system_one_product_hf.XXXXXX)
printf '[system-one-product-start] staging pinned model\n'
cp -a "$shared_cache/models--Qwen--Qwen3-0.6B" "$model_cache/"
export HF_HUB_CACHE="$model_cache"

printf '[system-one-product-start] starting validation rollout limit=%s expected=%s legality_backoff=%s\n' "$limit" "$expected" "$legality_backoff"
python scripts/eval_system_one_product_start_pilot.py \
  --data-dir "$source_data" \
  --split valid \
  --route-checkpoint "$route_checkpoint" \
  --route-run "$route_run" \
  --typed-checkpoint "$typed_checkpoint" \
  --output "$output" \
  --limit "$limit" \
  --seed 17 \
  --max-actions 12 \
  --log-every "$log_every" \
  "${backoff_args[@]}"
test -s "$output/report.json"
test -s "$output/cases.jsonl"
python - "$output/report.json" "$output/cases.jsonl" "$expected" "$legality_backoff" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
case_count = sum(1 for line in Path(sys.argv[2]).open() if line.strip())
expected = int(sys.argv[3])
legality_backoff = bool(int(sys.argv[4]))
assert report["split"] == "valid"
assert report["evaluated_reactions"] == case_count == expected
assert report["reaction_denominator"] == 1319
assert report["legality_backoff"] is legality_backoff
print({"phase": "validated_report", "evaluated_reactions": case_count,
       "endpoint_exact": report["endpoint_exact"]}, flush=True)
PY
