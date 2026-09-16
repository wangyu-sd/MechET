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
echo "[meteor] starting verified anchor-branch RL; stdout/stderr remain in default POD logs"
exec python -u scripts/run_anchor_branch_rl.py \
  --config "${1:-configs/agent/python_anchor_branch_rl_a100.yaml}"
