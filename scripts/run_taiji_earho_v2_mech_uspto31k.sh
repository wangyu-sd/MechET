#!/usr/bin/env bash
set -Eeuo pipefail

code_dir=/aaa/fionafyang/buddy1/whaleywang/MechET-nl-v2-full-runtime-20260918-02
artifact_root=/aaa/fionafyang/buddy1/whaleywang/MechET
vllm_ceph=$artifact_root/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu124_py311
vllm_archive=$artifact_root/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu124_py311_pruned.tar.zst
config=${1:-$code_dir/configs/agent/earho_v2_mech_uspto31k_8a100.yaml}
if [[ $# -gt 0 ]]; then shift; fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$code_dir"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_CACHE=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

if [[ ! -f "$vllm_ceph/.mechet_vllm_runtime_complete" ]]; then
  echo "[earho-v2] incomplete vLLM runtime: $vllm_ceph" >&2
  exit 2
fi
if [[ ! -f "$vllm_archive" ]]; then
  echo "[earho-v2] missing single-file vLLM runtime archive: $vllm_archive" >&2
  exit 2
fi
wheel_target=$(mktemp -d /tmp/mechet_earho_v2_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheel_target" \
  "$artifact_root/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl" \
  "$artifact_root/artifacts/wheels/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl" \
  "$artifact_root/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl" \
  "$artifact_root/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
export PYTHONPATH=$wheel_target:$code_dir/src:$code_dir

python - <<'PY'
import os, re, rdkit, torch
names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
expected = os.environ.get('MECHET_EXPECTED_GPU_REGEX', 'A100')
if len(names) != 8 or not all(re.search(expected, name) for name in names):
    raise SystemExit(f'expected ordinary 8x{expected}, got {names}')
if rdkit.__version__ != '2026.03.4':
    raise SystemExit(f'expected RDKit 2026.03.4, got {rdkit.__version__}')
print({'hardware': names, 'rdkit': rdkit.__version__}, flush=True)
PY

vllm_local=$(mktemp -d /tmp/mechet_earho_v2_vllm.XXXXXX)
echo "[earho-v2] extracting pinned vLLM 0.8.5 archive to $vllm_local"
tar --zstd -xf "$vllm_archive" -C "$vllm_local"
test -f "$vllm_local/.mechet_vllm_runtime_complete"
export MECHET_ANCHOR_VLLM_RUNTIME=$vllm_local

echo "[earho-v2] code=$(git rev-parse HEAD) config=$config"
exec python -u scripts/run_earho_v2.py --config "$config" "$@"
