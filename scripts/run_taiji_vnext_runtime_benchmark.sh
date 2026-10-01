#!/usr/bin/env bash
set -Eeuo pipefail

repo=${MECHET_AUTORESEARCH_CODE_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET-autoresearch-vnext-20261001}
artifact_root=${MECHET_ARTIFACT_ROOT:-/aaa/fionafyang/buddy1/whaleywang/MechET}
data=${VNEXT_RUNTIME_DATA:-$artifact_root/data/mech_uspto_31k_natural_language_history_v2/valid.jsonl}
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
# Eight independent vLLM workers share the host CPUs. Without this bound,
# each graph-capture warmup spawns the host-wide default (~124 threads) while
# constructing dummy LoRA weights, and all eight workers contend indefinitely.
export OMP_NUM_THREADS="${VNEXT_CPU_THREADS_PER_WORKER:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
[[ "$OMP_NUM_THREADS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[vnext-runtime] invalid VNEXT_CPU_THREADS_PER_WORKER=$OMP_NUM_THREADS" >&2
  exit 2
}
echo "[vnext-runtime] cpu_threads_per_worker=$OMP_NUM_THREADS" >&2
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

python - <<PY
import hashlib, json
from pathlib import Path
source = Path("$data")
manifest = json.loads((source.parent / "manifest.json").read_text())
assert source.is_file() and manifest["splits"]["valid"]["event_decisions"] >= 128
assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["splits"]["valid"]["output_sha256"]
print({"phase": "data_gate", "event_decisions": manifest["splits"]["valid"]["event_decisions"]}, flush=True)
PY

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
worker_pids=()
stop_workers() {
  for pid in "${worker_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap stop_workers TERM INT HUP
for gpu in {0..7}; do
  echo "[vnext-runtime] start shard=$gpu visible_gpu=$gpu" >&2
  CUDA_VISIBLE_DEVICES="$gpu" VNEXT_RANK="$gpu" VNEXT_WORLD_SIZE=8 \
    python -u scripts/benchmark_vnext_structured_vllm.py \
      --data "$data" --adapter "$adapter" --output "$output/raw" \
      --count 128 --max-new-tokens 384 \
      --modes eager_prefix graph_no_prefix graph_prefix graph_schema graph_inventory &
  worker_pids+=("$!")
done
worker_failed=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then worker_failed=1; fi
done
trap - TERM INT HUP
(( worker_failed == 0 )) || { echo "[vnext-runtime] at least one shard failed" >&2; exit 1; }

python scripts/summarize_vnext_runtime_benchmark.py \
  --input-dir "$output/raw" --output "$output/summary.json" \
  --expected-ranks 8 --expected-states 128 \
  --expected-modes eager_prefix graph_no_prefix graph_prefix graph_schema graph_inventory

echo "[vnext-runtime] complete: $output/summary.json" >&2
