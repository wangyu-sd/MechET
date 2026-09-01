#!/usr/bin/env bash
set -Eeuo pipefail

# Build an isolated CUDA-12 vLLM runtime in a versioned shared Ceph cache.  The
# legacy Taiji MechET image carries torch cu118 but the A100 hosts expose a
# CUDA-12 driver/runtime.  Keeping the dependency set outside the environment
# avoids mutation, preserves node-local space for model staging, and makes a
# failed compatibility check explicit.
runtime_dir=${MECHET_VLLM_RUNTIME_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu124_py311}
marker="$runtime_dir/.mechet_vllm_runtime_complete"
stage_dir="${runtime_dir}.stage.$$"
python_bin=/root/miniconda3/envs/meteor/bin/python
index_url=${MECHET_PYPI_INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple}

if [[ ! -f "$marker" ]]; then
  rm -rf "$stage_dir"
  mkdir -p "$stage_dir"
  echo "[meteor-vllm-bootstrap] stage=install version=0.8.5 destination=$runtime_dir time=$(date --iso-8601=seconds)"
  "$python_bin" -m pip install \
    --disable-pip-version-check \
    --ignore-installed \
    --target "$stage_dir" \
    --index-url "$index_url" \
    'vllm==0.8.5' \
    'transformers==4.55.4'
  PYTHONPATH="$stage_dir" "$python_bin" - <<'PY'
import torch
import vllm

assert vllm.__version__ == "0.8.5", vllm.__version__
assert torch.version.cuda and int(torch.version.cuda.split(".", 1)[0]) >= 12, torch.version.cuda
assert torch.cuda.is_available(), "CUDA is unavailable"
major, minor = torch.cuda.get_device_capability(0)
assert major >= 8, f"vLLM A100 runtime requires Ampere+, got capability {major}.{minor}"
print(
    {
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": f"{major}.{minor}",
    },
    flush=True,
)
PY
  printf 'vllm=0.8.5\n' > "$stage_dir/.mechet_vllm_runtime_complete"
  rm -rf "$runtime_dir"
  mv "$stage_dir" "$runtime_dir"
fi

export PYTHONPATH="$runtime_dir${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/envs/meteor/bin:$PATH"
export MECHET_INFERENCE_BACKEND=vllm
export MECHET_SKIP_CONDA_ACTIVATE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONUNBUFFERED=1

"$python_bin" - <<'PY'
import torch
import vllm
print(
    f"[meteor-vllm-ready] backend=vllm version={vllm.__version__} "
    f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}",
    flush=True,
)
PY

exec bash /aaa/fionafyang/buddy1/whaleywang/MechET/scripts/run_taiji_flower_a7_inference.sh
