#!/usr/bin/env bash
set -Eeuo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
repo_dir=/aaa/fionafyang/buddy1/whaleywang/MechET
cd "$repo_dir"
export HF_HUB_CACHE=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$repo_dir/src:$repo_dir/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Importing the isolated vLLM tree from Ceph in eight processes at once causes
# a metadata storm and can leave every rank in uninterruptible I/O sleep while
# all GPUs remain idle.  Stage it once on node-local storage, then let all
# workers share the local page cache.
ceph_vllm_runtime=${MECHET_CEPH_VLLM_RUNTIME:-$repo_dir/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu124_py311}
if [[ ! -f "$ceph_vllm_runtime/.mechet_vllm_runtime_complete" ]]; then
  echo "[meteor] missing complete Ceph vLLM runtime: $ceph_vllm_runtime" >&2
  exit 2
fi
local_vllm_runtime=$(mktemp -d /tmp/meteor-anchor-vllm.XXXXXX)
echo "[meteor] staging vLLM runtime from Ceph to $local_vllm_runtime"
cp -a "$ceph_vllm_runtime"/. "$local_vllm_runtime"/
test -f "$local_vllm_runtime/.mechet_vllm_runtime_complete"
export MECHET_ANCHOR_VLLM_RUNTIME="$local_vllm_runtime"
echo "[meteor] local vLLM runtime ready: $MECHET_ANCHOR_VLLM_RUNTIME"
echo "[meteor] starting verified anchor-branch RL; stdout/stderr remain in default POD logs"
exec python -u scripts/run_anchor_branch_rl.py \
  --config "${1:-configs/agent/python_anchor_branch_rl_a100.yaml}"
