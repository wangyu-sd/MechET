#!/usr/bin/env bash
# Stage III of PR #82; caller must supply the frozen Stage-II adapter SHA-256.
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=/aaa/fionafyang/buddy1/whaleywang/MechET-flower-stage3-full-20261008
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
vllm_ceph=$shared_repo/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu124_py311
stage2=$shared_repo/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17
template=$runtime_repo/configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml
: "${MECHET_RELIABLE_STAGEII_SHA256:?freeze the Stage-II adapter SHA-256 before EARHO}"
echo "$MECHET_RELIABLE_STAGEII_SHA256  $stage2/adapter_model.safetensors" | sha256sum --check --strict
test -s "$stage2/adapter_manifest.json"
test -f "$vllm_ceph/.mechet_vllm_runtime_complete"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
cd "$runtime_repo"
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$shared_hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MECHET_RELIABLE_DATA_ROOT=$shared_repo

wheel_target=$(mktemp -d /tmp/mechet_reliable_earho_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheel_target" \
  "$shared_repo/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl" \
  "$shared_repo/artifacts/wheels/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl" \
  "$shared_repo/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl" \
  "$shared_repo/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
export PYTHONPATH=$wheel_target:$runtime_repo/src:$runtime_repo

python - <<'PY'
import rdkit, torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names) == 8 and all('A100' in name.upper() for name in names), names
assert rdkit.__version__ == '2026.03.4', rdkit.__version__
print({'gate':'passed','gpus':names,'rdkit':rdkit.__version__}, flush=True)
PY

vllm_local=$(mktemp -d /tmp/mechet_reliable_earho_vllm.XXXXXX)
echo "[reliable-earho] staging frozen vLLM runtime to $vllm_local"
cp -a "$vllm_ceph/." "$vllm_local/"
test -f "$vllm_local/.mechet_vllm_runtime_complete"
export MECHET_ANCHOR_VLLM_RUNTIME=$vllm_local

runtime_config_dir=$(mktemp -d /tmp/mechet_reliable_earho_config.XXXXXX)
runtime_config=$runtime_config_dir/earho.yaml
python - "$template" "$runtime_config" "$MECHET_RELIABLE_STAGEII_SHA256" <<'PY'
import sys, yaml
from pathlib import Path
source, target, digest = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
cfg = yaml.safe_load(source.read_text())
assert cfg['initial_adapter_model_sha256'] == 'REPLACE_WITH_FROZEN_STAGE_II_SHA256'
cfg['initial_adapter_model_sha256'] = digest
target.write_text(yaml.safe_dump(cfg, sort_keys=False))
print({'stage':'earho-config-frozen','parent_sha256':digest,'config':str(target)}, flush=True)
PY

code_commit=$(git -C "$runtime_repo" rev-parse HEAD)
code_dir=$(mktemp -d /tmp/mechet_reliable_earho_code.XXXXXX)
git -C "$runtime_repo" archive "$code_commit" | tar -x -C "$code_dir"
export PYTHONPATH=$wheel_target:$code_dir/src:$code_dir
cd "$code_dir"
echo "[reliable-earho] immutable_code_commit=$code_commit runtime=$runtime_config"
exec python -u scripts/run_earho_v2.py --config "$runtime_config"
