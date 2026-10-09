#!/usr/bin/env bash
# V100/FP16 variant of the frozen FlowER Stage-III EARHO run.
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
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE=$shared_hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MECHET_RELIABLE_DATA_ROOT=$shared_repo MECHET_EARHO_ASYNC_REACTIONS=4
export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS

wheel_target=$(mktemp -d /tmp/mechet_reliable_earho_v100_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheel_target" \
  "$shared_repo/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl" \
  "$shared_repo/artifacts/wheels/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl" \
  "$shared_repo/artifacts/wheels/bitsandbytes-0.49.2-py3-none-manylinux_2_24_x86_64.whl" \
  "$shared_repo/artifacts/wheels/rdkit-2026.3.4-cp311-cp311-manylinux_2_28_x86_64.whl"
export PYTHONPATH=$wheel_target:$runtime_repo/src:$runtime_repo

python - <<'PY'
import rdkit, torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names) == 8 and all('V100' in name.upper() for name in names), names
assert rdkit.__version__ == '2026.03.4', rdkit.__version__
print({'gate': 'v100-host-passed', 'gpus': names, 'rdkit': rdkit.__version__}, flush=True)
PY

vllm_local=$(mktemp -d /tmp/mechet_reliable_earho_v100_vllm.XXXXXX)
echo "[reliable-earho-v100] staging frozen vLLM runtime to $vllm_local"
cp -a "$vllm_ceph/." "$vllm_local/"
test -f "$vllm_local/.mechet_vllm_runtime_complete"
export MECHET_ANCHOR_VLLM_RUNTIME=$vllm_local
PYTHONPATH=$vllm_local:$PYTHONPATH python - <<'PY'
import torch, vllm
assert vllm.__version__ == '0.8.5', vllm.__version__
assert torch.cuda.is_available(), 'CUDA runtime cannot drive allocated V100'
assert torch.cuda.get_device_capability(0)[0] == 7
print({'gate': 'v100-vllm-import-passed', 'vllm': vllm.__version__,
       'torch_cuda': torch.version.cuda}, flush=True)
PY

runtime_config_dir=$(mktemp -d /tmp/mechet_reliable_earho_v100_config.XXXXXX)
runtime_config=$runtime_config_dir/earho.yaml
python - "$template" "$runtime_config" "$MECHET_RELIABLE_STAGEII_SHA256" <<'PY'
import sys, yaml
from pathlib import Path
source, target, digest = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
cfg = yaml.safe_load(source.read_text())
assert cfg['initial_adapter_model_sha256'] == 'REPLACE_WITH_FROZEN_STAGE_II_SHA256'
cfg['initial_adapter_model_sha256'] = digest
cfg['expected_gpu_regex'] = 'V100'
cfg['rollout']['dtype'] = 'float16'
cfg['output_dir'] = 'outputs/agent/earho_reliable_mechet_qwen3_0_6b_full_executable_v100_seed17'
target.write_text(yaml.safe_dump(cfg, sort_keys=False))
print({'stage': 'earho-v100-config-frozen', 'parent_sha256': digest,
       'precision': 'float16', 'output': cfg['output_dir']}, flush=True)
PY

code_commit=$(git -C "$runtime_repo" rev-parse HEAD)
code_dir=$(mktemp -d /tmp/mechet_reliable_earho_v100_code.XXXXXX)
git -C "$runtime_repo" archive "$code_commit" | tar -x -C "$code_dir"
grep -q 'class AsyncVLLMBridge' "$code_dir/scripts/natural_language_anchor_branch_stage.py"
grep -q 'train_dtype' "$code_dir/scripts/python_continual_stage.py"
export PYTHONPATH=$wheel_target:$code_dir/src:$code_dir
cd "$code_dir"
echo "[reliable-earho-v100] immutable_code_commit=$code_commit runtime=$runtime_config async_reactions_per_gpu=$MECHET_EARHO_ASYNC_REACTIONS"
exec python -u scripts/run_earho_v2.py --config "$runtime_config"
