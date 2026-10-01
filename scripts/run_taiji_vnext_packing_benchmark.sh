#!/usr/bin/env bash
set -Eeuo pipefail

repo=${MECHET_AUTORESEARCH_CODE_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-autoresearch-vnext-20261001}
artifact_root=${MECHET_ARTIFACT_ROOT:-/aaa/fionafyang/buddy1/whaleywang/MechET}
data=${VNEXT_PACKING_DATA:-$artifact_root/data/mech_uspto_31k_natural_language_history_v2/train.jsonl}
adapter=${VNEXT_POLICY_ADAPTER:-$artifact_root/outputs/agent/mech_uspto31k_nl_history_v2_qwen3_8b_h20_seed17_20260922}
output=${VNEXT_PACKING_OUTPUT:-$artifact_root/outputs/autoresearch/vnext_packing_a100_20261001/summary.json}
model_cache=${VNEXT_MODEL_CACHE:-/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache}

cd "$repo"
if [[ -n "${MECHET_EXPECTED_CODE_COMMIT:-}" ]]; then
  actual=$(git rev-parse HEAD)
  [[ "$actual" == "$MECHET_EXPECTED_CODE_COMMIT" ]] || {
    echo "[vnext-packing] code mismatch expected=$MECHET_EXPECTED_CODE_COMMIT actual=$actual" >&2
    exit 2
  }
fi
[[ ! -e "$output" ]] || { echo "[vnext-packing] refusing existing output $output" >&2; exit 2; }

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo/src:$repo"

python - <<'PY'
import torch
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print({"phase":"gpu_gate","names":names}, flush=True)
assert len(names)>=1 and "A100" in names[0].upper()
PY

staged_cache=$(mktemp -d /tmp/mechet_vnext_packing_hf.XXXXXX)
cp -a "$model_cache/models--Qwen--Qwen3-8B" "$staged_cache/"
export HF_HUB_CACHE="$staged_cache"

python scripts/benchmark_vnext_packing.py   --data "$data" --adapter "$adapter" --output "$output"   --rows 32 --packs 4 --max-length 4096 --warmup 1

echo "[vnext-packing] complete: $output" >&2
