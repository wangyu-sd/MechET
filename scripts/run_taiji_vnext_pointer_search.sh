#!/usr/bin/env bash
set -Eeuo pipefail

repo=${MECHET_AUTORESEARCH_CODE_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-autoresearch-vnext-20261001}
artifact_root=${MECHET_ARTIFACT_ROOT:-/aaa/fionafyang/buddy1/whaleywang/MechET}
source_data=${VNEXT_SOURCE_DATA:-$artifact_root/data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/valid.jsonl}
adapter=${VNEXT_POLICY_ADAPTER:-$artifact_root/outputs/agent/mech_uspto31k_nl_history_v2_qwen3_8b_h20_seed17_20260922}
pointer=${VNEXT_POINTER_HEAD:-$artifact_root/outputs/agent/pr71_conditional_pointer_31k_stage2_8a100_20261001/pointer_head_epoch1.pt}
output_base=${VNEXT_OUTPUT_BASE:?VNEXT_OUTPUT_BASE is required}
sample_reactions=${VNEXT_SAMPLE_REACTIONS:?VNEXT_SAMPLE_REACTIONS is required}
model_cache=${VNEXT_MODEL_CACHE:-/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache}

cd "$repo"
if [[ -n "${MECHET_EXPECTED_CODE_COMMIT:-}" ]]; then
  actual=$(git rev-parse HEAD)
  if [[ "$actual" != "$MECHET_EXPECTED_CODE_COMMIT" ]]; then
    echo "[vnext-autoresearch] code mismatch expected=$MECHET_EXPECTED_CODE_COMMIT actual=$actual" >&2
    exit 2
  fi
fi
if [[ -e "$output_base" ]]; then
  echo "[vnext-autoresearch] refusing existing output: $output_base" >&2
  exit 2
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$repo/src:$repo"

python - <<'PY'
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print({"phase": "gpu_gate", "gpu_count": len(names), "gpu_names": names}, flush=True)
assert len(names) == 8 and all("A100" in name.upper() for name in names)
PY

wheels=$(mktemp -d /tmp/mechet_vnext_ar_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheels"   "$artifact_root/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
export PYTHONPATH="$wheels:$PYTHONPATH"

python - <<PY
import hashlib, json
from pathlib import Path
source = Path("$source_data")
manifest = json.loads((source.parent / "manifest.json").read_text())
assert source.is_file() and Path("$pointer").is_file()
assert manifest["splits"]["valid"]["rows"] == 1319
assert manifest["validation_passed"] and manifest["tokenizer_audit_passed"]
assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["splits"]["valid"]["sha256"]
print({"phase": "data_gate", "sample_reactions": int("$sample_reactions")}, flush=True)
PY

staged_cache=$(mktemp -d /tmp/mechet_vnext_ar_hf.XXXXXX)
cp -a "$model_cache/models--Qwen--Qwen3-8B" "$staged_cache/"
export HF_HUB_CACHE="$staged_cache"

common=(--data "$source_data" --policy-adapter "$adapter" --model Qwen/Qwen3-8B
  --sample-reactions "$sample_reactions" --seed 17 --reaction-level --compact-history
  --vnext-v2-prefix --no-4bit --search-no-value --value-weight 0
  --early-beam 1 --late-beam 1 --max-decisions 40
  --max-imports 32 --max-new-tokens 384)

if [[ "${VNEXT_INCLUDE_GREEDY:-0}" == "1" ]]; then
  echo "[vnext-autoresearch] greedy K=1" >&2
  torchrun --standalone --nproc_per_node=8 scripts/run_natural_language_value_search.py     "${common[@]}" --branching 1 --output "$output_base/greedy_k1"
fi

echo "[vnext-autoresearch] matched K=4 policy baseline" >&2
torchrun --standalone --nproc_per_node=8 scripts/run_natural_language_value_search.py   "${common[@]}" --branching 4 --output "$output_base/baseline_k4"

echo "[vnext-autoresearch] matched K=4 coupled-pointer reranking" >&2
torchrun --standalone --nproc_per_node=8 scripts/run_natural_language_value_search.py   "${common[@]}" --branching 4 --pointer-head "$pointer" --pointer-weight 0.2   --output "$output_base/coupled_pointer_k4"

python scripts/summarize_vnext_pointer_search.py   --baseline "$output_base/baseline_k4/results.shard-*-of-08.jsonl"   --pointer "$output_base/coupled_pointer_k4/results.shard-*-of-08.jsonl"   --expected-rows "$sample_reactions"   --output "$output_base/paired_summary.json"

echo "[vnext-autoresearch] completed sample_reactions=$sample_reactions output=$output_base" >&2
