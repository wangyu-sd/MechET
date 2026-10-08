#!/usr/bin/env bash
# User-authorized State-SFT -> EARHO direct condition; no Stage-II history prompt.
set -Eeuo pipefail

shared=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime=/aaa/fionafyang/buddy1/whaleywang/MechET-pr82-reliable-20261006
hf=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
vllm=$shared/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu124_py311
config=$runtime/configs/agent/earho_mech_uspto31k_state_direct_06b_a100.yaml
parent=$shared/outputs/agent/reliable_mech_uspto31k_state_qwen3_0_6b_v100_seed17
echo '1eb152a34f417e289fb2fb3e05a2d1a79fb2cc7a99eac8dbfd168cce8bb2c7e8  '"$parent/adapter_model.safetensors" | sha256sum --check --strict
test -s "$parent/adapter_manifest.json"
test -f "$vllm/.mechet_vllm_runtime_complete"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$runtime"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_CACHE=$hf TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

wheel_target=$(mktemp -d /tmp/mechet_31k_direct_earho_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheel_target" \
  "$shared/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl" \
  "$shared/artifacts/wheels/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl" \
  "$shared/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl" \
  "$shared/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
export PYTHONPATH=$wheel_target:$runtime/src:$runtime

python - <<'PY'
import rdkit, torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names) == 8 and all('A100' in name.upper() for name in names), names
assert rdkit.__version__ == '2026.03.4', rdkit.__version__
print({'gate':'hardware_passed','gpus':names,'rdkit':rdkit.__version__},flush=True)
PY

vllm_local=$(mktemp -d /tmp/mechet_31k_direct_earho_vllm.XXXXXX)
echo "[31k-direct-earho] staging vLLM runtime"
cp -a "$vllm/." "$vllm_local/"
export MECHET_ANCHOR_VLLM_RUNTIME=$vllm_local
echo "[31k-direct-earho] code=$(git rev-parse HEAD) config=$config"
exec python -u scripts/run_earho_v2.py --config "$config"
