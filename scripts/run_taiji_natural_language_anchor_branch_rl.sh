#!/usr/bin/env bash
set -Eeuo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
code_dir=/aaa/fionafyang/buddy1/whaleywang/MechET-nl-anchor-branch-rl-20260916
artifact_root=/aaa/fionafyang/buddy1/whaleywang/MechET
cd "$code_dir"
export HF_HUB_CACHE=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$code_dir/src:$code_dir/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ceph_vllm_runtime=${MECHET_CEPH_VLLM_RUNTIME:-$artifact_root/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu124_py311}
if [[ ! -f "$ceph_vllm_runtime/.mechet_vllm_runtime_complete" ]]; then
  echo "[meteor] missing complete Ceph vLLM runtime: $ceph_vllm_runtime" >&2
  exit 2
fi
local_vllm_runtime=$(mktemp -d /tmp/meteor-nl-anchor-vllm.XXXXXX)
echo "[meteor] staging vLLM runtime to $local_vllm_runtime"
cp -a "$ceph_vllm_runtime"/. "$local_vllm_runtime"/
test -f "$local_vllm_runtime/.mechet_vllm_runtime_complete"
export MECHET_ANCHOR_VLLM_RUNTIME="$local_vllm_runtime"
echo "[meteor] starting natural-language anchor-branch post-training"
echo "[meteor] parent=natural_language_event_sft_qwen3_8b_a100_seed17_20260913"
exec python -u scripts/run_natural_language_anchor_branch_rl.py \
  --config "${1:-configs/agent/natural_language_anchor_branch_rl_smoke_a100.yaml}"
