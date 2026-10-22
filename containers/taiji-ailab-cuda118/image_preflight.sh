#!/usr/bin/env bash
set -Eeuo pipefail

minimum_glibc=2.28
glibc_version=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
lowest_version=$(printf '%s\n%s\n' "$minimum_glibc" "$glibc_version" | sort -V | head -n 1)

echo "[meteor-image] os=$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release | tr -d '\"')"
echo "[meteor-image] glibc=$glibc_version required_glibc_min=$minimum_glibc"
echo "[meteor-image] nvidia_require_cuda=${NVIDIA_REQUIRE_CUDA:-unset}"
echo "[meteor-image] contract=${MECHET_IMAGE_CONTRACT:-unset}"

if [[ "$lowest_version" != "$minimum_glibc" ]]; then
  echo "[meteor-image][error] glibc $glibc_version is older than $minimum_glibc" >&2
  exit 21
fi

if [[ ${1:-} == --build ]]; then
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[meteor-image][error] nvidia-smi is unavailable" >&2
  exit 22
fi

nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader

python_bin=${MECHET_PYTHON_BIN:-/aaa/fionafyang/buddy1/whaleywang/miniconda3/envs/meteor/bin/python}
if [[ ! -x "$python_bin" ]]; then
  echo "[meteor-image][error] Python environment is missing: $python_bin" >&2
  exit 23
fi

"$python_bin" -u - <<'PY'
import json

import pyarrow
import sklearn
import torch
import transformers
import peft
from rdkit import Chem

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.version.cuda == "11.8", torch.version.cuda
assert Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O") is not None

x = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all().item()

print(json.dumps({
    "stage": "ailab-image-runtime-ok",
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "pyarrow": pyarrow.__version__,
    "sklearn": sklearn.__version__,
    "gpu": torch.cuda.get_device_name(0),
}), flush=True)
PY
