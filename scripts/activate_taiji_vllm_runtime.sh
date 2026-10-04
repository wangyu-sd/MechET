#!/usr/bin/env bash
set -Eeuo pipefail

# Reuse the frozen, already validated Taiji vLLM wheel set.  Keep the image and
# its Conda environment unchanged; only this child process receives the
# versioned Ceph runtime on PYTHONPATH.
runtime_dir=${MECHET_VLLM_RUNTIME_DIR:-/aaa/fionafyang/buddy1/whaleywang/MechET/artifacts/taiji_vllm_runtime/vllm_0_8_5_torch_2_6_cu124_py311}
marker="$runtime_dir/.mechet_vllm_runtime_complete"
if [[ ! -f "$marker" ]]; then
  echo >&2 "[meteor-vllm] missing frozen runtime marker: $marker"
  exit 2
fi
if [[ $# -eq 0 ]]; then
  echo >&2 "[meteor-vllm] expected a child command"
  exit 2
fi

export PYTHONPATH="$runtime_dir${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/envs/meteor/bin:$PATH"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONUNBUFFERED=1

/root/miniconda3/envs/meteor/bin/python - <<'PY'
import torch
import vllm

if vllm.__version__ != "0.8.5":
    raise SystemExit(f"expected vLLM 0.8.5, got {vllm.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("frozen vLLM runtime cannot see CUDA")
print(
    f"[meteor-vllm-ready] version={vllm.__version__} "
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"gpu={torch.cuda.get_device_name(0)}",
    flush=True,
)
PY

exec "$@"
