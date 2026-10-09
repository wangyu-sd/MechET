#!/usr/bin/env bash
# V100/FP16 variant of the frozen FlowER Stage-III EARHO run.
set -Eeuo pipefail

shared_repo=/aaa/fionafyang/buddy1/whaleywang/MechET
runtime_repo=/aaa/fionafyang/buddy1/whaleywang/MechET-flower-stage3-full-20261008
shared_hf_cache=/aaa/fionafyang/buddy1/whaleywang/OpenEvolveChem/data/hf_cache
vllm_archive=$shared_repo/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu118_py311.tar.zst
dependency_archive=$shared_repo/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu124_py311_pruned.tar.zst
stage2=$shared_repo/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17
template=$runtime_repo/configs/agent/earho_reliable_mechet_qwen3_0_6b.yaml
: "${MECHET_RELIABLE_STAGEII_SHA256:?freeze the Stage-II adapter SHA-256 before EARHO}"
echo "$MECHET_RELIABLE_STAGEII_SHA256  $stage2/adapter_model.safetensors" | sha256sum --check --strict
test -s "$stage2/adapter_manifest.json"
test -s "$vllm_archive"
test -s "$dependency_archive"
echo "23c44e907d6f50ed75f46f908063a3d6c37144d676dd444cc055df9a93b98f9c  $vllm_archive" | sha256sum --check --strict

source /root/miniconda3/etc/profile.d/conda.sh
conda activate meteor
export PYTHONUNBUFFERED=1
export PATH=$PATH:/root/miniconda3/bin
export HF_HUB_CACHE=$shared_hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MECHET_RELIABLE_DATA_ROOT=$shared_repo MECHET_EARHO_ASYNC_REACTIONS=4
export MECHET_EARHO_PREP_WORKERS=16
export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS

wheel_target=$(mktemp -d /tmp/mechet_reliable_earho_v100_wheels.XXXXXX)
python -m pip install --quiet --no-deps --target "$wheel_target" \
  "$shared_repo/artifacts/wheels/liger_kernel-0.6.2-py3-none-any.whl" \
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
fallback_local=$(mktemp -d /tmp/mechet_reliable_earho_v100_dependencies.XXXXXX)
echo "[reliable-earho-v100] staging CUDA 11.8 vLLM to $vllm_local"
tar --zstd -xf "$vllm_archive" -C "$vllm_local"
# This source archive also contains CUDA 12 packages. Extract only its
# non-CUDA dependencies; torch/vLLM/xFormers come from CUDA 11.8 builds.
tar --zstd --exclude='./torch*' --exclude='./functorch*' \
  --exclude='./nvidia*' --exclude='./triton*' --exclude='./vllm*' \
  --exclude='./xformers*' --exclude='./cupy*' --exclude='./cupyx*' \
  --exclude='./cuda*' -xf "$dependency_archive" -C "$fallback_local"
cp "$runtime_repo/scripts/earho_cuda11_sitecustomize.py" "$vllm_local/sitecustomize.py"
touch "$vllm_local/.mechet_vllm_runtime_complete"
export MECHET_ANCHOR_VLLM_RUNTIME=$vllm_local
export MECHET_VLLM_PUREPY_FALLBACK=$fallback_local
PYTHONPATH=$vllm_local:$PYTHONPATH python - <<'PY'
import torch, vllm, vllm._C, xformers
assert vllm.__version__ == '0.8.5', vllm.__version__
assert xformers.__version__ == '0.0.29.post2', xformers.__version__
assert torch.cuda.is_available(), 'CUDA runtime cannot drive allocated V100'
assert torch.cuda.get_device_capability(0)[0] == 7
assert torch.version.cuda == '11.8', torch.version.cuda
print({'gate': 'v100-vllm-import-passed', 'vllm': vllm.__version__,
       'xformers': xformers.__version__, 'torch_cuda': torch.version.cuda}, flush=True)
PY

runtime_config_dir=$(mktemp -d /tmp/mechet_reliable_earho_v100_config.XXXXXX)
runtime_config=$runtime_config_dir/earho.yaml
python - "$template" "$runtime_config" "$MECHET_RELIABLE_STAGEII_SHA256" "$vllm_local" <<'PY'
import sys, yaml
from pathlib import Path
source, target, digest, runtime = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
cfg = yaml.safe_load(source.read_text())
assert cfg['initial_adapter_model_sha256'] == 'REPLACE_WITH_FROZEN_STAGE_II_SHA256'
cfg['initial_adapter_model_sha256'] = digest
cfg['expected_gpu_regex'] = 'V100'
cfg['vllm_runtime'] = runtime
cfg['rollout']['dtype'] = 'float16'
cfg['output_dir'] = 'outputs/agent/earho_reliable_mechet_qwen3_0_6b_full_executable_parallel_v100_seed17'
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
echo "[reliable-earho-v100] immutable_code_commit=$code_commit runtime=$runtime_config async_reactions_per_gpu=$MECHET_EARHO_ASYNC_REACTIONS prep_workers=$MECHET_EARHO_PREP_WORKERS"
exec python -u scripts/run_earho_v2.py --config "$runtime_config"
