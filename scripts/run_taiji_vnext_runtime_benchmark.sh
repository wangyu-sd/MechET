#!/usr/bin/env bash
set -Eeuo pipefail

repo=${MECHET_AUTORESEARCH_CODE_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-autoresearch-vnext-20261001}
artifact_root=${MECHET_ARTIFACT_ROOT:-/aaa/fionafyang/buddy1/whaleywang/MechET}
data=${VNEXT_RUNTIME_DATA:-$artifact_root/data/mech_uspto_31k_inverse_tool_sft_action_delta_v2_compiler_20260824/valid.jsonl}
adapter=${VNEXT_POLICY_ADAPTER:-$artifact_root/outputs/agent/mech_uspto31k_nl_history_v2_qwen3_8b_h20_seed17_20260922}
output=${VNEXT_RUNTIME_OUTPUT:-$artifact_root/outputs/autoresearch/vnext_runtime_a100_20261001}
model_cache=${VNEXT_MODEL_CACHE:-/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache}
ceph_vllm_runtime=${MECHET_CEPH_VLLM_RUNTIME:-$artifact_root/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu124_py311}

cd "$repo"
if [[ -n "${MECHET_EXPECTED_CODE_COMMIT:-}" ]]; then
  actual=$(git rev-parse HEAD)
  [[ "$actual" == "$MECHET_EXPECTED_CODE_COMMIT" ]] || {
    echo "[vnext-runtime] code mismatch expected=$MECHET_EXPECTED_CODE_COMMIT actual=$actual" >&2
    exit 2
  }
fi
[[ ! -e "$output" ]] || { echo "[vnext-runtime] refusing existing output $output" >&2; exit 2; }

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="$repo/src:$repo"

python - <<'PY'
import torch
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print({"phase":"gpu_gate","names":names}, flush=True)
assert len(names)==8 and all("A100" in x.upper() for x in names)
PY

wheels=$(mktemp -d /tmp/mechet_vnext_runtime_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheels"   "$artifact_root/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
export PYTHONPATH="$wheels:$PYTHONPATH"

[[ -f "$ceph_vllm_runtime/.mechet_vllm_runtime_complete" ]] || {
  echo "[vnext-runtime] missing pinned vLLM runtime" >&2; exit 2;
}
local_vllm=$(mktemp -d /tmp/mechet_vnext_vllm.XXXXXX)
cp -a "$ceph_vllm_runtime"/. "$local_vllm"/
export PYTHONPATH="$local_vllm:$PYTHONPATH"

staged_cache=$(mktemp -d /tmp/mechet_vnext_runtime_hf.XXXXXX)
cp -a "$model_cache/models--Qwen--Qwen3-8B" "$staged_cache/"
export HF_HUB_CACHE="$staged_cache"

mkdir -p "$output"
torchrun --standalone --nproc_per_node=8 scripts/benchmark_vnext_structured_vllm.py   --data "$data" --adapter "$adapter" --output "$output/raw"   --count 128 --max-new-tokens 384   --modes eager_prefix graph_no_prefix graph_prefix graph_schema graph_inventory

python scripts/summarize_vnext_runtime_benchmark.py   --input-dir "$output/raw" --output "$output/summary.json"

echo "[vnext-runtime] complete: $output/summary.json" >&2
