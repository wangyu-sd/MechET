#!/usr/bin/env bash
set -Eeuo pipefail
shared=/aaa/fionafyang/buddy1/whaleywang/MechET
repo=/aaa/fionafyang/buddy1/whaleywang/MechET-pr91-rsi
: "${MECHET_RSI_CODE_SHA:?immutable PR91 code required}"
mountpoint -q /aaa/fionafyang/buddy1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
runtime=$(mktemp -d /tmp/mechet_rsi_code.XXXXXX)
git -C "$repo" archive "$MECHET_RSI_CODE_SHA" | tar -x -C "$runtime"
vllm_runtime=$(mktemp -d /tmp/mechet_rsi_vllm.XXXXXX)
bash "$runtime/scripts/unpack_taiji_runtime.sh" \
 "$shared/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu118_py311.tar.gz" \
 b1185493fb316b2d4534d888df2a6cd871dbb74f6c95255235ba4947ae8486a5 "$vllm_runtime"
deps=$(mktemp -d /tmp/mechet_rsi_deps.XXXXXX)
bash "$runtime/scripts/unpack_taiji_runtime.sh" \
 "$shared/artifacts/taiji_vllm_runtime/vllm_0_8_5_noncuda_dependencies_from_archive_20261014.tar.gz" \
 817efd049ae42972a4be38b1b43c9f975fb068050c8f52923cfcc85178bd389d "$deps"
cp "$runtime/scripts/vllm_cuda11_sitecustomize.py" "$vllm_runtime/sitecustomize.py"
rdkit_local=$(mktemp -d /tmp/mechet_rsi_rdkit.XXXXXX)
wheel=$shared/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl
echo "a41dde42ecb24e7d93d62b89e8e5d267b02bbac764f965f3968296c99234c685  $wheel" | sha256sum --check --strict
python -m pip install --no-deps --no-index --target "$rdkit_local" "$wheel"
export PYTHONPATH=$rdkit_local:$vllm_runtime:$runtime/src:$runtime
export MECHET_VLLM_PUREPY_FALLBACK=$deps
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_P2P_DISABLE=1
local_hf=$(mktemp -d /tmp/mechet_rsi_hf.XXXXXX)
cp -aL /aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache/models--Qwen--Qwen3-0.6B "$local_hf/"
export HF_HUB_CACHE=$local_hf
cd "$runtime"
python -u - <<'PY'
import torch,rdkit,vllm
assert torch.cuda.device_count()==2
assert all('A100' in torch.cuda.get_device_name(i) for i in range(2))
assert torch.version.cuda=='11.8' and rdkit.__version__=='2026.03.4'
assert vllm.__version__.split('+')[0]=='0.8.5'
print({'phase':'PR91-runtime','torch':torch.__version__,'vllm':vllm.__version__,'rdkit':rdkit.__version__,
       'gpus':[torch.cuda.get_device_name(i) for i in range(2)]},flush=True)
PY
python -u scripts/run_mechet_rsi_smoke.py --config configs/experiments/mechet_rsi_resolved_20261021.yaml
